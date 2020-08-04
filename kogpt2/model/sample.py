import torch
import torch.nn.functional as F


def top_k_logits(logits, k):
    if k == 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1]
    return torch.where(logits < min_values, torch.ones_like(logits, dtype=logits.dtype) * -1e10, logits)


def top_p_logits(logits, top_p=0.0, filter_value=-float('Inf')):
    """Nucleus sampling"""
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs >= top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[:, indices_to_remove] = filter_value
    return logits


def sample_sequence(model, tok, vocab, sent, text_size, temperature, top_p, top_k):
    ctx = 'cuda'
    device = torch.device(ctx)

    toked = tok(sent) # 받은 문장
    count = 0
    generated_text = ''

    if len(toked) > 1024:
        return 0

    while 1:  # 이부분도 적절하게 바꾸기.
        # 시작 토큰 넣기
        input_ids = torch.tensor([vocab[vocab.bos_token], ] + vocab[toked]).unsqueeze(0)

        input_ids = input_ids.to(ctx)
        model = model.to(ctx)

        predicts = model(input_ids)
        pred = predicts[0]

        # temperature 적용
        logits = pred
        logits = logits[:, -1, :] / temperature
        # top k
        logits = top_k_logits(logits, top_k)
        # top p
        logits = top_p_logits(logits, top_p=top_p)

        #logits = logits.to(ctx)

        # 확률적을 뽑고
        log_probs = F.softmax(logits, dim=-1)
        # 이전 것들 저장해서 다음 학습에 사용
        prev = torch.multinomial(log_probs, num_samples=1)
        # 결과 나오게 (사전에서 gpt2가 뽑은 결과)
        gen = vocab.to_tokens(prev.squeeze().tolist())

        # 끝나면 본격적으로 만들어 놓기.
        if gen == '</s>' or count > text_size:
            print(count)
            print('to_tokens:', vocab.to_tokens(torch.argmax(pred, axis=-1).squeeze().tolist()))
            #print(sent)
            sent += gen.replace('▁', ' ')
            generated_text += gen.replace('▁', ' ')
            sent += '\n'
            generated_text += '\n'
            toked = tok(sent)
            count = 0
            break

        sent += gen.replace('▁', ' ')
        generated_text += gen.replace('▁', ' ')
        toked = tok(sent)
        count += 1
    return sent