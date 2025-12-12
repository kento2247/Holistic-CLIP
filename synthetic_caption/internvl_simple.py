import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import json


def collate_fn(batch):
    indices = [item['idx'] for item in batch]
    original_captions = [item['original_caption'] for item in batch]
    pixel_values_list = [item['pixel_values'] for item in batch]
    num_patches_list = [pv.size(0) for pv in pixel_values_list]
    pixel_values = torch.cat(pixel_values_list, dim=0)
    return {
        'indices': indices,
        'original_captions': original_captions,
        'pixel_values': pixel_values,
        'num_patches_list': num_patches_list
    }

class CC3MDataset(Dataset):
    def __init__(self, split='all', input_size=448, max_num=6, start_idx=0, end_idx=-1):
        self.input_size = input_size
        self.max_num = max_num
        if split == 'all':
            self.dataset = load_dataset('cc3m', split='train').concat(
                load_dataset('cc3m', split='validation')
            ).concat(
                load_dataset('cc3m', split='test')
            )
        else:
            self.dataset = load_dataset('cc3m', split=split)
        # slice dataset
        self.dataset = self.dataset.select(range(start_idx, end_idx)) if end_idx != -1 else self.dataset.select(range(start_idx, len(self.dataset)))

        self.IMAGENET_MEAN = (0.485, 0.456, 0.406)
        self.IMAGENET_STD = (0.229, 0.224, 0.225)

    def __len__(self):
        return len(self.dataset)

    def _build_transform(self, input_size):
        return T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
        ])


    def _find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    
    def _dynamic_preprocess(self,image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1)
            for i in range(1, n + 1) for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
        target_aspect_ratio = self._find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images




    def _process_image(self,image, input_size=448, max_num=6):
        """Process PIL Image to pixel values."""
        image = image.convert('RGB')
        transform = self._build_transform(input_size=input_size)
        images = self._dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(img) for img in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values.to(torch.bfloat16)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        pixel_values = self._process_image(image, self.input_size, self.max_num)
        return {
            'idx': idx,
            'pixel_values': pixel_values,
            'original_caption': item.get('txt', '')
        }

class InternVLCaptioner:
    def __init__(
        self,
        model_path='OpenGVLab/InternVL2-8B',
        device='cuda',
        max_new_tokens=77,
        prompt='<image>\nDescribe the image in detail.'
    ):
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.prompt = prompt

        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to(device).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer.padding_side = 'left'

        self.generation_config = dict(
            num_beams=1,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    def caption_batch(self, pixel_values, num_patches_list):
        pixel_values = pixel_values.to(self.device)
        question_list = [self.prompt] * len(num_patches_list)

        queries, eos_token_id, template_sep = self.model.question2query(
            question_list, num_patches_list=num_patches_list, tokenizer=self.tokenizer
        )
        model_inputs = self.tokenizer(queries, return_tensors='pt', padding=True)
        model_inputs['input_ids'] = model_inputs['input_ids'].to(self.device)
        model_inputs['attention_mask'] = model_inputs['attention_mask'].to(self.device)

        responses = self.model.batch_chat(
            self.tokenizer, pixel_values,
            num_patches_list=num_patches_list,
            questions=question_list,
            generation_config=self.generation_config,
            device=self.device,
            model_inputs=model_inputs,
            eos_token_id=eos_token_id,
            template_sep=template_sep
        )
        return responses


def main(args):
    dataset = CC3MDataset(split=args.split,start_idx=args.start_idx,end_idx=args.end_idx)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )

    captioner = InternVLCaptioner(
        model_path=args.model_path,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        prompt=args.prompt
    )

    results = []
    for batch in tqdm(dataloader):
        responses = captioner.caption_batch(
            batch['pixel_values'],
            batch['num_patches_list']
        )
        for idx, original_caption, synthetic_caption in zip(
            batch['indices'], batch['original_captions'], responses
        ):
            results.append({
                'idx': idx,
                'original_caption': original_caption,
                'synthetic_caption': synthetic_caption
            })

    with open(args.output_json, 'w') as f:
        json.dump(results, f, indent=2)

    return results


def arg_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_json', type=str, default='cc3m_captions.json')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'validation', 'test', 'all'])
    parser.add_argument('--model_path', type=str, default='OpenGVLab/InternVL2-8B')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max_new_tokens', type=int, default=77)
    parser.add_argument('--prompt', type=str, default='<image>\nDescribe the image in detail.')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=-1)
    return parser.parse_args()

if __name__ == "__main__":
    args = arg_parser()
    main(args)
