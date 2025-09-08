export MSYS_NO_PATHCONV=1
PATH_SUBSET=out/subset
generate:
	@: torchのダウンロードにめっちゃ時間がかかるがデッドロックはしておらず、待てば終わる
	pipenv install
generate-subset:
	python ./pick_flatten_subset_copy.py --src synfashion_release --dst $(PATH_SUBSET) --total 2000
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
run-negative:
	# pipenv run accelerate config
	curl -LO https://raw.githubusercontent.com/huggingface/diffusers/c9c82173068d628b0569ccb6d656adfa37a389e8/examples/textual_inversion/textual_inversion.py
	pipenv run accelerate launch \
	--config_file "accelerate_configuration.yaml" \
	textual_inversion.py \
	--pretrained_model_name_or_path=runwayml/stable-diffusion-v1-5 \
	--train_data_dir $(PATH_SUBSET) \
	--placeholder_token="neg_fashion" \
	--initializer_token="style" \
	--output_dir=./out
clean:
	pipenv --rm