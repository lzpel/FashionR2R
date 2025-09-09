from diffusers import StableDiffusionPipeline
import torch, torch.nn.functional

def tokenid(dir: str, token: str):
	pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
	pipe.load_textual_inversion(dir, token=token)
	tid = pipe.tokenizer.convert_tokens_to_ids(token)
	print("token_id:", tid, "(unk? ->", pipe.tokenizer.unk_token_id, ")")
	emb = pipe.text_encoder.get_input_embeddings().weight.detach()
	print("norm(neg_fashion):", float(emb[tid].norm()))
def cos(dir: str, token: str):
	pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
	tid_style = pipe.tokenizer.convert_tokens_to_ids("style")
	pipe.load_textual_inversion(dir, token=token)
	tid = pipe.tokenizer.convert_tokens_to_ids(token)
	W = pipe.text_encoder.get_input_embeddings().weight.detach()
	cos = torch.nn.functional.cosine_similarity(W[tid].unsqueeze(0), W[tid_style].unsqueeze(0)).item()
	print("cos(neg_fashion, style) =", cos)
if __name__ == "__main__":
	tokenid("./out", "neg_fashion")
	cos("./out", "neg_fashion")