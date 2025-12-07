from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, BitsAndBytesConfig
import torch
import torch.nn.functional as F
import numpy as np


def strip_double_quotes(input_str):
    if input_str.startswith('"') and input_str.endswith('"'):
        return input_str[1:-1]
    return input_str


MODEL_DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
BASE_MODEL_KWARGS = {"torch_dtype": MODEL_DTYPE, "device_map": "auto"}
QUANTIZATION_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=MODEL_DTYPE,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)


class HuggingFaceLanguageModel:
    def __init__(self, repo_name: str, combine_system_user=False, should_quantize=False, token=None):
        """
        Initialize the Hugging Face model class in a distributed manner.

        Args:
            repo_name (str): Name of the Hugging Face model repository, e.g., "meta-llama/Meta-Llama-3-8B".
            token (str): Hugging Face API token for private models.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(repo_name, token=token)

        model_kwargs = BASE_MODEL_KWARGS.copy()
        if should_quantize:
            model_kwargs["quantization_config"] = QUANTIZATION_CONFIG
        print(f"Loading language model {repo_name} with config {model_kwargs}")
        self.model = AutoModelForCausalLM.from_pretrained(repo_name, token=token, **model_kwargs)
        self.model.eval()

        self.combine_system_user = combine_system_user

    def generate(self, system: str, user: str, **kwargs):
        """
        Generate a response based on the input text.

        Args:
            system (str): System message for the model.
            user (str): User message for the model.
            max_length (int): Maximum length of the generated response.
            **kwargs: Additional optional parameters such as temperature, top_k, top_p.

        Returns:
            str: The generated response from the model.
        """
        messages = self._get_message_turns(system, user)
        plain_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Model and tokenizer will handle device placement automatically
        inputs = self.tokenizer(plain_text, return_tensors="pt")
        # Move inputs to the correct device based on their device_map
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        outputs = self.model.generate(
            **inputs,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **kwargs,
        )
        response_start = inputs["input_ids"].shape[-1]
        response_ids = outputs[0][response_start:]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        response = strip_double_quotes(response)
        return response

    def continue_generate(self, system: str, user1: str, assistant1: str, user2: str, max_length: int = 1000, **kwargs):
        """
        Continue a conversation and generate a response.

        Args:
            system (str): System message for the model.
            user1 (str): User message for the model.
            assistant1 (str): Assistant message for the model.
            user2 (str): User message for the model.
            max_length (int): Maximum length of the generated response.
            **kwargs: Additional optional parameters such as temperature, top_k, top_p.

        Returns:
            str: The generated response from the model.
        """
        messages = [
            {'role': 'system', 'content': f'{system}'},
            {'role': 'user', 'content': f'{user1}'},
            {'role': 'assistant', 'content': f'{assistant1}'},
            {'role': 'user', 'content': f'{user2}'},
        ]
        plain_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self.tokenizer(plain_text, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **kwargs,
        )
        response_start = inputs["input_ids"].shape[-1]
        response_ids = outputs[0][response_start:]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        response = strip_double_quotes(response)
        return response

    def conditional_generate(self, condition: str, system: str, user: str, max_length: int = 1000, **kwargs):
        """
        Generate a response with additional conditions appended to the input prompt.

        Args:
            condition (str): Condition for the generation (appended to the prompt).
            system (str): System message for the model.
            user (str): User message for the model.
            max_length (int): Maximum length of the generated response.
            **kwargs: Additional optional parameters such as temperature, top_k, top_p.

        Returns:
            str: The generated response from the model.
        """
        messages = self._get_message_turns(system, user)
        plain_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        plain_text += condition

        inputs = self.tokenizer(plain_text, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **kwargs,
        )
        response_start = inputs["input_ids"].shape[-1]
        response_ids = outputs[0][response_start:]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        response = strip_double_quotes(response)
        return response

    def _get_message_turns(self, system, user):
        if self.combine_system_user:
            return [{"role": "user", "content": f"[SYSTEM]: {system}\n\n[USER]: {user}"}]
        else:
            return [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]


class HuggingFaceEmbeddingModel:
    def __init__(
            self,
            repo_name: str,
            pooling_strategy: str,
            max_length: int,
            embed_instruction: str,
            should_quantize=False,
            token=None,
        ):
        """
        Initialize the Hugging Face model class in a distributed manner.

        Args:
            repo_name (str): Name of the Hugging Face model repository, e.g., "meta-llama/Meta-Llama-3-8B".
            token (str): Hugging Face API token for private models.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(repo_name, token=token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.pooling_strategy = pooling_strategy
        self.max_length = max_length
        self.embed_instruction = embed_instruction

        model_kwargs = BASE_MODEL_KWARGS.copy()
        if should_quantize:
            model_kwargs["quantization_config"] = QUANTIZATION_CONFIG
        print(f"Loading embedding model {repo_name} with config {model_kwargs}")
        self.model = AutoModel.from_pretrained(repo_name, token=token, **model_kwargs)
        self.model.eval()

    @torch.no_grad()
    def encode(self, text):
        single_input = False
        if isinstance(text, str):
            text = [text]
            single_input = True

        text = [self.embed_instruction.format(query=query) for query in text]

        # Tokenize the input texts
        batch_dict = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch_dict.to(self.model.device)
        outputs = self.model(**batch_dict)
        embeddings = self.apply_pooling(outputs.last_hidden_state, batch_dict['attention_mask'])

        # normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)

        embeddings = embeddings.float().cpu().numpy().astype(np.float32)

        if single_input and len(embeddings) == 1:
            return embeddings[0]
        return embeddings

    def apply_pooling(self, last_hidden_states, attention_mask):
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])

        if self.pooling_strategy == "last":
            if left_padding:
                return last_hidden_states[:, -1]
            else:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = last_hidden_states.shape[0]
                return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
