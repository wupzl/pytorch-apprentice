"""Chapter 13: shifted response-only SFT and classic DPO loss checks."""
import torch
import torch.nn.functional as F


def sft_loss(logits, labels):
    # labels are UNSHIFTED; mask prompt/padding with -100 before this call.
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    if not (shift_labels != -100).any():
        raise ValueError('Batch contains no supervised response tokens')
    return F.cross_entropy(shift_logits.reshape(-1, shift_logits.size(-1)),
                           shift_labels.reshape(-1), ignore_index=-100)


def response_logps(logits, labels):
    # Sum conditional token log probabilities over response positions only.
    targets = labels[:, 1:]
    mask = targets != -100
    if not mask.any(dim=-1).all():
        raise ValueError('Each response must contain a supervised token')
    safe_targets = targets.masked_fill(~mask, 0)
    token_logps = logits[:, :-1].log_softmax(-1).gather(
        -1, safe_targets.unsqueeze(-1)
    ).squeeze(-1)
    return (token_logps * mask).sum(-1)


def dpo_loss(policy_chosen_logp, policy_rejected_logp,
             ref_chosen_logp, ref_rejected_logp, beta=0.1):
    if beta <= 0:
        raise ValueError('beta must be positive')
    policy_logratio = policy_chosen_logp - policy_rejected_logp
    ref_logratio = (ref_chosen_logp - ref_rejected_logp).detach()
    return -F.logsigmoid(beta * (policy_logratio - ref_logratio)).mean()


def main():
    torch.manual_seed(42)
    # Tokens 1,2,3 form the prompt; 4,5 are the response, 0 is PAD.
    labels = torch.tensor([[-100, -100, -100, 4, 5, -100]])
    logits = torch.randn(1, 6, 8, requires_grad=True)
    loss = sft_loss(logits, labels)
    loss.backward()
    assert logits.grad[:, :2].abs().sum() == 0
    assert logits.grad[:, 2:4].abs().sum() > 0  # includes prediction of first answer token
    assert logits.grad[:, 4:].abs().sum() == 0
    manual = logits.detach()[0, 2].log_softmax(-1)[4] + logits.detach()[0, 3].log_softmax(-1)[5]
    assert torch.allclose(response_logps(logits.detach(), labels)[0], manual)
    try:
        sft_loss(logits, torch.full_like(labels, -100))
    except ValueError:
        pass
    else:
        raise AssertionError('All-masked targets must be rejected')
    chosen = torch.tensor([-3.0], requires_grad=True)
    rejected = torch.tensor([-4.0], requires_grad=True)
    ref_chosen = torch.tensor([-3.0], requires_grad=True)
    ref_rejected = torch.tensor([-4.0], requires_grad=True)
    dpo = dpo_loss(chosen, rejected, ref_chosen, ref_rejected)
    assert torch.allclose(dpo, torch.log(torch.tensor(2.0)))
    dpo.backward()
    assert chosen.grad.item() < 0 < rejected.grad.item()
    assert ref_chosen.grad is None and ref_rejected.grad is None
    improved = dpo_loss(chosen.detach() + 1, rejected.detach(), ref_chosen, ref_rejected)
    assert improved < dpo
    print('PASS: SFT alignment/masking, response log-probs, DPO reference and gradient direction.')


if __name__ == '__main__':
    main()
