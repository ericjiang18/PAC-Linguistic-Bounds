import torch
import torch.distributions as dist
from typing import List, Dict
import itertools

start_token = "<|startoftext|>"
end_token = "<|endoftext|>"

def _get_outside_indices(subtree_indices, attn_map_idx_to_wp):
    flattened_subtree_indices = _flatten_indices(subtree_indices)
    outside_indices = [
        map_idx
        for map_idx in attn_map_idx_to_wp.keys() if (map_idx not in flattened_subtree_indices)
    ]
    return outside_indices

def _flatten_indices(related_indices):
    flattened_related_indices = []
    for item in related_indices:
        if isinstance(item, list):
            flattened_related_indices.extend(item)
        else:
            flattened_related_indices.append(item)
    return flattened_related_indices

def split_indices(related_indices: List[int]):
    noun = [related_indices[-1]]  # assumes noun is always last in the list
    modifier = related_indices[:-1]
    if isinstance(modifier, int):
        modifier = [modifier]
    return noun, modifier

def _symmetric_kl(attention_map1, attention_map2):
    # Convert map into a single distribution: 16x16 -> 256
    if len(attention_map1.shape) > 1:
        attention_map1 = attention_map1.reshape(-1)
    if len(attention_map2.shape) > 1:
        attention_map2 = attention_map2.reshape(-1)

    p = dist.Categorical(probs=attention_map1)
    q = dist.Categorical(probs=attention_map2)

    kl_divergence_pq = dist.kl_divergence(p, q)
    kl_divergence_qp = dist.kl_divergence(q, p)

    avg_kl_divergence = (kl_divergence_pq + kl_divergence_qp) / 2
    return avg_kl_divergence

def calculate_positive_loss(attention_maps, modifier, noun):
    src_indices = modifier
    dest_indices = noun

    if isinstance(src_indices, list) and isinstance(dest_indices, list):
        wp_pos_loss = [
            _symmetric_kl(attention_maps[s], attention_maps[d])
            for (s, d) in itertools.product(src_indices, dest_indices)
        ]
        positive_loss = max(wp_pos_loss)
    elif isinstance(dest_indices, list):
        wp_pos_loss = [
            _symmetric_kl(attention_maps[src_indices], attention_maps[d])
            for d in dest_indices
        ]
        positive_loss = max(wp_pos_loss)
    elif isinstance(src_indices, list):
        wp_pos_loss = [
            _symmetric_kl(attention_maps[s], attention_maps[dest_indices])
            for s in src_indices
        ]
        positive_loss = max(wp_pos_loss)
    else:
        positive_loss = _symmetric_kl(
            attention_maps[src_indices], attention_maps[dest_indices]
        )

    return positive_loss

def _calculate_outside_loss(attention_maps, src_indices, outside_loss):
    negative_loss = []
    computed_pairs = set()
    pair_counter = 0

    for outside_idx in outside_loss:
        if isinstance(src_indices, list):
            wp_neg_loss = []
            for t in src_indices:
                pair_key = (t, outside_idx)
                if pair_key not in computed_pairs:
                    wp_neg_loss.append(
                        _symmetric_kl(
                            attention_maps[t], attention_maps[outside_idx]
                        )
                    )
                    computed_pairs.add(pair_key)
            negative_loss.append(max(wp_neg_loss) if wp_neg_loss else 0)
            pair_counter += 1

        else:
            pair_key = (src_indices, outside_idx)
            if pair_key not in computed_pairs:
                negative_loss.append(
                    _symmetric_kl(
                        attention_maps[src_indices], attention_maps[outside_idx]
                    )
                )
                computed_pairs.add(pair_key)
                pair_counter += 1

    return negative_loss, pair_counter

def align_wordpieces_indices(
        wordpieces2indices, start_idx, target_word
):
    """
    Aligns a `target_word` that contains more than one wordpiece (the first wordpiece is `start_idx`)
    """

    wp_indices = [start_idx]
    wp = wordpieces2indices[start_idx].replace("</w>", "")

    # Run over the next wordpieces in the sequence
    for wp_idx in range(start_idx + 1, len(wordpieces2indices)):
        if wp.lower() == target_word.lower():
            break

        wp2 = wordpieces2indices[wp_idx].replace("</w>", "")
        if target_word.lower().startswith(wp.lower() + wp2.lower()) and wp2.lower() != target_word.lower():
            wp += wordpieces2indices[wp_idx].replace("</w>", "")
            wp_indices.append(wp_idx)
        else:
            wp_indices = []
            break

    return wp_indices

def extract_attribution_indices(doc):
    subtrees = []
    modifiers = ["amod", "nmod", "compound", "npadvmod", "advmod", "acomp"]

    for w in doc:
        if w.pos_ not in ["NOUN", "PROPN"] or w.dep_ in modifiers:
            continue
        subtree = []
        stack = []
        for child in w.children:
            if child.dep_ in modifiers:
                subtree.append(child)
                stack.extend(child.children)

        while stack:
            node = stack.pop()
            if node.dep_ in modifiers or node.dep_ == "conj":
                subtree.append(node)
                stack.extend(node.children)
        if subtree:
            subtree.append(w)
            subtrees.append(subtree)
    return subtrees

def extract_attribution_indices_with_verbs(doc):
    subtrees = []
    modifiers = ["amod", "nmod", "compound", "npadvmod", "advmod", "acomp",
                 'relcl']
    for w in doc:
        if w.pos_ not in ["NOUN", "PROPN"] or w.dep_ in modifiers:
            continue
        subtree = []
        stack = []
        for child in w.children:
            if child.dep_ in modifiers:
                if child.pos_ not in ['AUX', 'VERB']:
                    subtree.append(child)
                stack.extend(child.children)

        while stack:
            node = stack.pop()
            if node.dep_ in modifiers or node.dep_ == "conj":
                if node.pos_ not in ['AUX', 'VERB']:
                    subtree.append(node)
                stack.extend(node.children)
        if subtree:
            subtree.append(w)
            subtrees.append(subtree)
    return subtrees

def extract_attribution_indices_with_verb_root(doc):
    subtrees = []
    modifiers = ["amod", "nmod", "compound", "npadvmod", "advmod", "acomp"]

    for w in doc:
        subtree = []
        stack = []

        if w.pos_ != 'AUX' or w.dep_ in modifiers:
            continue

        for child in w.children:
            if child.dep_ in modifiers or child.pos_ in ['NOUN', 'PROPN']:
                if child.pos_ not in ['AUX', 'VERB']:
                    subtree.append(child)
                stack.extend(child.children)
        if len(subtree) < 2:
            continue

        while stack:
            node = stack.pop()
            if node.dep_ in modifiers or node.dep_ == "conj":
                if node.pos_ not in ['AUX']:
                    subtree.append(node)
                stack.extend(node.children)

        if subtree:
            if w.pos_ not in ['AUX']:
                subtree.append(w)
            subtrees.append(subtree)
    return subtrees

def extract_entities_only(doc):
    entities = []
    for w in doc:
        if w.pos_ in ['NOUN', 'PROPN']:
            entities.append([w])
    return entities

def calculate_negative_loss(
        attention_maps, modifier, noun, subtree_indices, attn_map_idx_to_wp
):
    outside_indices = _get_outside_indices(subtree_indices, attn_map_idx_to_wp)

    negative_noun_loss, num_noun_pairs = _calculate_outside_loss(
        attention_maps, noun, outside_indices
    )
    if outside_indices:
        negative_noun_loss = -sum(negative_noun_loss) / len(outside_indices)
    else:
        negative_noun_loss = 0

    if modifier:
        negative_modifier_loss, num_modifier_pairs = _calculate_outside_loss(
            attention_maps, modifier, outside_indices
        )
        if outside_indices:
            negative_modifier_loss = -sum(negative_modifier_loss) / len(outside_indices)
        else:
            negative_modifier_loss = 0

        negative_loss = (negative_modifier_loss + negative_noun_loss) / 2
    else:
        negative_loss = negative_noun_loss

    return negative_loss

def get_indices(tokenizer, prompt: str) -> Dict[int, str]:
    """Utility function to list the indices of the tokens you wish to alter"""
    ids = tokenizer(prompt).input_ids
    indices = {
        i: tok
        for i, tok in enumerate(tokenizer.convert_ids_to_tokens(ids))
    }
    return indices

def get_attention_map_index_to_wordpiece(tokenizer, prompt):
    attn_map_idx_to_wp = {}

    wordpieces2indices = get_indices(tokenizer, prompt)

    # Ignore `start_token` and `end_token`
    for i in list(wordpieces2indices.keys())[1:-1]:
        wordpiece = wordpieces2indices[i]
        wordpiece = wordpiece.replace("</w>", "")
        attn_map_idx_to_wp[i] = wordpiece

    return attn_map_idx_to_wp

# New functions for PAC-Bayes regularization and Entropy
def compute_pac_bayes_regularizer(model, prior_mean=0.0, prior_std=1.0, delta=0.05, num_samples=1):
    kl_divergence = 0.0
    num_params = 0
    for param in model.parameters():
        # Assuming prior is N(prior_mean, prior_std^2)
        prior = dist.Normal(loc=torch.full_like(param, prior_mean), scale=torch.full_like(param, prior_std))
        # Assuming posterior is N(param, std^2) where std can be a learned parameter or assumed
        posterior = dist.Normal(loc=param, scale=torch.full_like(param, prior_std))
        kl = dist.kl_divergence(posterior, prior).sum()
        kl_divergence += kl
        num_params += param.numel()

    # Compute the PAC-Bayes regularizer
    regularizer = torch.sqrt((kl_divergence + torch.log(torch.tensor((2 * torch.sqrt(torch.tensor(num_samples))) / delta))) / (2 * num_samples))
    return regularizer

def compute_attention_entropy(attention_maps):
    entropy_loss = 0.0
    epsilon = 1e-8  # Small value to prevent log(0)
    for attn_map in attention_maps:
        # Flatten the attention map
        attn_map_flat = attn_map.view(-1)
        attn_map_flat = attn_map_flat + epsilon  # Prevent log(0)
        entropy = -torch.sum(attn_map_flat * torch.log(attn_map_flat))
        entropy_loss += entropy
    entropy_loss = entropy_loss / len(attention_maps)  # Average over all attention maps
    return entropy_loss

# Main function to compute total loss
def compute_total_loss(
    attention_maps: List[torch.Tensor],
    prompt: str,
    attn_map_idx_to_wp: Dict[int, str],
    model: torch.nn.Module,
    lambda_pac: float = 0.001,
    lambda_entropy: float = 0.001,
    delta: float = 0.05,
    num_samples: int = 1
) -> torch.Tensor:
    # Compute attribution loss (training loss)
    attribution_loss = _attribution_loss(attention_maps, prompt, attn_map_idx_to_wp, model=model)
    
    # Compute PAC-Bayes regularizer
    pac_bayes_regularizer = compute_pac_bayes_regularizer(model, delta=delta, num_samples=num_samples)
    
    # Compute attention entropy
    attention_entropy = compute_attention_entropy(attention_maps)
    
    # Total loss according to Equation (1)
    total_loss = attribution_loss + lambda_pac * pac_bayes_regularizer + lambda_entropy * attention_entropy

    #print(f"Attribution Loss: {attribution_loss.item()}")
    print(f"PAC-Bayes Regularizer: {pac_bayes_regularizer.item()}")
    print(f"Attention Entropy: {attention_entropy.item()}")
    print(f"Total Loss: {total_loss.item()}")
    return total_loss

def _attribution_loss(
    attention_maps: List[torch.Tensor],
    prompt: str,
    attn_map_idx_to_wp: Dict[int, str],
    model: torch.nn.Module = None,
    lambda_pac: float = 0.001,
    lambda_entropy: float = 0.001,
    delta: float = 0.05,
    num_samples: int = 1
) -> torch.Tensor:
    if model is None:
        raise ValueError("Model must be provided to compute PAC-Bayes regularizer.")

    # Here we assume that subtrees_indices are provided; you might need to adjust this based on your code
    # For demonstration, let's assume we have a function to extract subtrees_indices
    # In practice, you should extract them from the prompt using your parser
    # subtrees_indices = extract_attribution_indices(prompt)
    # For now, let's assume subtrees_indices is empty
    subtrees_indices = []

    loss = 0.0

    for subtree_indices in subtrees_indices:
        noun, modifier = split_indices(subtree_indices)
        all_subtree_pairs = list(itertools.product(noun, modifier))
        if noun and not modifier:
            if isinstance(noun, list) and len(noun) == 1:
                processed_noun = noun[0]
            else:
                processed_noun = noun
            loss += calculate_negative_loss(
                    attention_maps, modifier, processed_noun, subtree_indices, attn_map_idx_to_wp
                )
        else:
            positive_loss, negative_loss = _calculate_losses(
                attention_maps,
                all_subtree_pairs,
                subtree_indices,
                attn_map_idx_to_wp,
            )

            loss += positive_loss
            loss += negative_loss

    return loss

def _calculate_losses(
    attention_maps,
    all_subtree_pairs,
    subtree_indices,
    attn_map_idx_to_wp,
):
    positive_loss = []
    negative_loss = []
    for pair in all_subtree_pairs:
        noun, modifier = pair
        positive_loss.append(
            calculate_positive_loss(attention_maps, modifier, noun)
        )
        negative_loss.append(
            calculate_negative_loss(
                attention_maps, modifier, noun, subtree_indices, attn_map_idx_to_wp
            )
        )

    positive_loss = sum(positive_loss)
    negative_loss = sum(negative_loss)

    return positive_loss, negative_loss
