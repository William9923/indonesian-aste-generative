# ## == Inference ==
# def inference(model, tokenizer, sents, fix=False):
    
#     # --- [Preprocessing] ---
#     splitted_sents = [txt.split(" ") for txt in sents]

#     # --- [Tokenization] ---
#     batch = tokenizer.batch_encode_plus(sents, max_length=MAX_LENGTH, padding='max_length', truncation=True,return_tensors="pt")
    
#     # --- [Generating Triplets Opinion] ---
#     model.eval()
#     outs = model.generate(input_ids=batch['input_ids'].to(device), 
#                                     attention_mask=batch['attention_mask'].to(device), 
#                                     max_length=MAX_LENGTH)
    
#     # --- [Decoding] ---
#     outputs = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
    
#     # --- [Normalization, if needed] ---
#     if fix:
#         all_preds = []
#         for out in outputs:
#             all_preds.append(extract(out))
#         outputs = fix_preds(all_preds, splitted_sents)
        
#     return outputs

# sents = [
#     "pelayanan ramah , kamar nyaman dan fasilitas lengkap . hanya airnya showernya kurang panas .",
#     "tidak terlalu jauh dari pusat kota .",
#     "dengan harga terjangkau kita sudah mendapatkan fasilitas yang nyaman .",
#     "kamar luas dan bersih . seprai bersih .",
#     "seprai nya kurang bersih .",
#     "kamarnya bersih dan rapi . saya kebetulan dapat yang di lantai dua ."
# ]

# labels = [
#     [([0], [1], 'POS'), ([3], [4], 'POS'), ([6], [7], 'POS'), ([10, 11], [12, 13], 'NEG')],
#     [([3, 4, 5], [0, 1, 2], 'POS')],
#     [([1], [2], 'POS'), ([6], [8], 'POS')],
#     [([0], [1], 'POS'), ([0], [3], 'POS'), ([0], [5], 'POS')],
#     [([0, 1], [2, 3], 'NEG')],
#     [([0], [1], 'POS'), ([0], [3], 'POS')],
# ] 

# res = inference(model, tokenizer, sents)

# # --- [Evaluate the example] ---
# targets = generate_extraction_style_target([sent.split(" ") for sent in sents], labels)
# raw_scores, fixed_scores, all_labels, all_preds, all_fixed_preds = evaluate(res, targets, [txt.split(" ") for txt in sents])
