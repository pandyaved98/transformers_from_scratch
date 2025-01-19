class Config:
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    dropout = 0.1
    max_seq_len = 512
    batch_size = 32
    num_epochs = 10
    learning_rate = 1e-4
    warmup_steps = 4000
    label_smoothing = 0.1
    vocab_size = 37000
    pad_token_id = 0
    attention_type = "vanilla"
    positional_embedding = "rotary"
    beam_size = 5
    length_penalty = 1.0
    use_fp16 = True
    use_gradient_checkpointing = True
    sparse_window_size = 32
    num_groups = 2
    use_dynamic_batching = True
