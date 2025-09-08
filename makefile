export MSYS_NO_PATHCONV=1
generate:
	@: torchのダウンロードにめっちゃ時間がかかるがデッドロックはしておらず、待てば終わる
	pipenv install
run:
	pipenv run $(MAKE) run-inside
run-inside:
	#--use_negEmbedding --negative_embedding_dir=#your trained negative domain embedding \
	CUDA_VISIBLE_DEVICES=0 python ./Realistic_translation.py \
	--model_path=runwayml/stable-diffusion-v1-5 \
	--source_image_path=synfashion_release \
	--output_dir=out \
	--source_prompt='' \
	--target_prompt='' \
	--replace_steps_ratio=0.9 \
	--denoising_strength=0.3 \
	--cfg_scale=7.5 \
	--attn_replace_layers=256 \
	--inversion_as_start
clean:
	pipenv --rm