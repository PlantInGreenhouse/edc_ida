# An example to run EDC (without refinement) on the example dataset

# OIE_LLM=mistralai/Mistral-7B-Instruct-v0.2
# SD_LLM=mistralai/Mistral-7B-Instruct-v0.2
# SC_LLM=mistralai/Mistral-7B-Instruct-v0.2
# SC_EMBEDDER=intfloat/e5-mistral-7b-instruct
OIE_LLM=gpt-4o
SD_LLM=gpt-4o
SC_LLM=gpt-4o
SC_EMBEDDER=text-embedding-ada-002
DATASET=example

python run.py \
    --oie_llm $OIE_LLM \
    --oie_few_shot_example_file_path "./few_shot_examples/example/oie_few_shot_examples.txt" \
    --sd_llm $SD_LLM \
    --sd_few_shot_example_file_path "./few_shot_examples/example/sd_few_shot_examples.txt" \
    --sc_llm $SC_LLM \
    --sc_embedder $SC_EMBEDDER \
    --input_text_file_path "./datasets/example.txt" \
    --target_schema_path "./schemas/example_schema.csv" \
    --output_dir "./output/test_target_alignment" \
    --logging_verbose \
    --refinement_iterations 1 \
    --ee_llm gpt-4o \
    --sr_embedder text-embedding-ada-002 \
    --ee_prompt_template_file_path "./prompt_templates/ee_template.txt" \
    --ee_few_shot_example_file_path "./few_shot_examples/example/ee_few_shot_examples.txt" \
    --em_prompt_template_file_path "./prompt_templates/em_template.txt"