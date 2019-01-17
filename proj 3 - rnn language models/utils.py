import torch


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_batch(sentences):
    min_len = min(map(len, sentences))
    input_sentences = []
    target_sentences = []
    for sentence in sentences:
        input_sentences.append(sentence[0: min_len - 1])
        target_sentences.append(sentence[1: min_len])

    target_sentences_combined = [sentence for sublist in target_sentences for sentence in sublist]
    return (torch.tensor(input_sentences, dtype=torch.long).t().to(get_device()),
            torch.tensor(target_sentences_combined, dtype=torch.long).to(get_device()))


def repackage_hidden(h):
    """
    Wrap hidden states in new Tensors, to detach them from their history.
    :param h:
    :return:
    """
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)
