import os 
import torch
import numpy as np
from diffusers import DDIMScheduler
from diffusers.utils import load_image
from utils.diffuser_utils import FashionR2RPipeline
from utils.Attention_replace import SelfAttentionControlEdit
from utils.attention_register_utils import register_attention_control_new
from torchvision.utils import save_image

import argparse
import time
def parse_args():
    parser = argparse.ArgumentParser(description = 'Example')
    parser.add_argument(
        "--model_path",
        type=str,
        default="",
        help="Basemodel path",
    )
    parser.add_argument(
        "--source_image_path",
        type=str,
        default=None,
        help="source_image_path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="output dir",
    )
    parser.add_argument(
        "--negative_embedding_dir",
        type=str,
        default=None,
        help="negative embedding dir",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default='<negative-embedding>',
        help="negative embedding tokens",
    )     
    parser.add_argument(
        "--source_prompt",
        type=str,
        default='',
        help="prompt of source image, if real image editing, this can be empty",
    )    
    parser.add_argument(
        "--target_prompt",
        type=str,
        default='',
        help="prompt for the target image",
    )    
    parser.add_argument(
        "--replace_steps_ratio",
        type=float,
        default=0.9,
        help='self attention replace steps, final steps = replace_steps_ratio * total_steps'
    )
    parser.add_argument(
        "--total_steps",
        type=int,
        default=50,
        help='Diffusion denoising steps'
    )
    parser.add_argument(
        "--denoising_strength",
        type=float,
        default=0.3,
        help='denoising_stength, same as the Denoising strength in diffusers pipeline'
    )
    parser.add_argument(
        "--attn_replace_layers",
        type=int,
        default=256,
        help='attention replace in which layers: HW/64 HW/64 HW/256 HW/256 HW/1024 HW/1024----HW/4096----HW/1024 HW/1024 HW/1024 HW/256 HW/256 HW/256 HW/64 HW/64 HW/64)'
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=7.5,
        help='Classifier free guidance scale'
    )
    parser.add_argument(
        "--inversion_as_start",
        action="store_true",
        help='Use inversion result as init noise or not'
    )
    parser.add_argument(
        "--use_negEmbedding",
        action="store_true",
        help='Use negative embedding or not'
    )
    parser.add_argument(
        "--use_attention_mask",
        action="store_true",
        help='Use mask in attention calculation or not'
    )
    args = parser.parse_args()
    return args

def save_args_to_sh(args, script_path):
    os.makedirs(os.path.dirname(script_path), exist_ok=True)

    with open(script_path, 'w') as file:
        file.write("#!/bin/bash\n\n")
        for arg in vars(args):
            file.write(f"export {arg.upper()}='{getattr(args, arg)}'\n")
        file.write("\n")
        file.write("CUDA_VISIBLE_DEVICES=2 python /path/to/your/script.py \\\n")
        for arg in vars(args):
            file.write(f"    --{arg}=${arg.upper()} \\\n")
        file.seek(file.tell() - 3)  # Remove the last backslash
        file.write("\n")
    
    os.chmod(script_path, 0o755)  # Make the script executable
    print(f"Script saved to {script_path}")
    

def main():
    flag = 0
    args = parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    
    pipe = FashionR2RPipeline.from_pretrained(args.model_path, scheduler=scheduler).to(device)

    for root, dirs, files in os.walk(args.source_image_path):
        for name in files:
            if name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                start_time = time.time()
                file_path = os.path.join(root, name)
                source_image = load_image(file_path)
                # source_image = source_image.crop((66, 88 , 66 + 768, 88 + 1024))
                source_image = source_image.resize((768,1024))
                H = source_image.size[1]
                W = source_image.size[0]
                if args.use_attention_mask:
                    print('using attention mask')
                    img_array = np.array(source_image)
                    mask = np.ones(img_array.shape[:2], dtype=np.uint8)
                    mask[(img_array == [0, 0, 0]).all(axis=2)] = 0  
                    mask[(img_array == [255, 255, 255]).all(axis=2)] = 0  
                    mask = torch.from_numpy(mask).unsqueeze(0)
                else:
                    mask = None

                layer_to_replace_flag = H * W / args.attn_replace_layers 


                # invert the source image
                start_code, latents_list = pipe.invert(source_image,
                                                       args.source_prompt,
                                                       guidance_scale=args.cfg_scale,
                                                       num_inference_steps=args.total_steps,
                                                       strength = args.denoising_strength,
                                                       return_intermediates=True)

                # negative prompt
                if args.use_negEmbedding and flag == 0:
                    print('Inversion Loaded')
                    print(args.use_negEmbedding)
                    pipe.load_textual_inversion(args.negative_embedding_dir,  token=args.negative_prompt)


                prompts = [args.source_prompt, args.target_prompt]

                start_code = start_code.expand(len(prompts), -1, -1, -1)
                controller = SelfAttentionControlEdit(prompts, args.total_steps * args.denoising_strength, self_replace_steps=args.replace_steps_ratio, flag = layer_to_replace_flag)

                register_attention_control_new(pipe, controller)

                pipe.enable_model_cpu_offload()
                # pipe.enable_xformers_memory_efficient_attention()

                generator = torch.Generator(device="cuda").manual_seed(1)

                results = pipe(prompts,
                               source_image,
                               latents=start_code,
                               guidance_scale=args.cfg_scale,
                               ref_intermediate_latents=latents_list,
                               height = H,
                               width = W,
                               strength = args.denoising_strength,
                               neg_prompt = args.negative_prompt,
                               generator = generator,
                               img_mask = mask,
                               inversion_as_start = args.inversion_as_start)
                target_subdir = os.path.relpath(root, args.source_image_path)
                target_folder = os.path.join(args.output_dir, target_subdir)
                os.makedirs(target_folder, exist_ok = True)

                save_image(results[1], os.path.join(target_folder, f"{H}x{W}_{name}_{args.target_prompt}_{args.replace_steps_ratio}_{args.denoising_strength}_attn{args.attn_replace_layers}.jpg"))
                target_file_path = os.path.join(target_folder, name)
                duration = time.time() - start_time
                print(f'image Saved to: {target_file_path}, time is {duration}')
                flag += 1
    save_args_to_sh(args, f'{target_folder}/run.sh')

if __name__=="__main__":
    main()