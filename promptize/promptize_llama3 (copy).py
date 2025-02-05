# Input and output file paths
"""
src_file = "en-de/news-commentary/news-commentary-v18.de-en.en"
tgt_file = "en-de/news-commentary/news-commentary-v18.de-en.de"
sco_file = "en-de/news-commentary/news-commentary-v18.de-en.cometkiwi22.sco"
output_file = "en-de/news-commentary/news-commentary-v18.de-en.prompted"

src_file = "en-de/cc-matrix/cc-matrix-ende.original.en"
tgt_file = "en-de/cc-matrix/cc-matrix-ende.original.de"
sco_file = "en-de/cc-matrix/cc-matrix-ende.original.cometkiwi22.sco"
output_file = "en-de/cc-matrix/cc-matrix-ende.original.prompted"

src_file = "en-de/europarl/europarl-v10.de-en.en"
tgt_file = "en-de/europarl/europarl-v10.de-en.de"
sco_file = "en-de/europarl/europarl-v10.de-en.cometkiwi22.sco"
output_file = "en-de/europarl/europarl-v10.de-en.prompted"
"""
src_file = "en-de/paracrawl/paracrawlv9.en-de.9M.en"
tgt_file = "en-de/paracrawl/paracrawlv9.en-de.9M.de"
sco_file = "en-de/paracrawl/paracrawlv9.en-de.9M.cometkiwiXL.sco"
output_file = "en-de/paracrawl/paracrawlv9.en-de.9M.prompted.TowerchatML"
"""
src_file = "en-de/bt_fromde/wikipedia/wikipedia-de.txt.clean.ppl13.en"
tgt_file = "en-de/bt_fromde/wikipedia/wikipedia-de.txt.clean.ppl13.de"
sco_file = "en-de/bt_fromde/wikipedia/wikipedia-de.txt.clean.ppl13.cometkiwi22.sco"
output_file = "en-de/bt_fromde/wikipedia/wikipedia-de.txt.clean.ppl13.prompted"

src_file = "en-de/bt_fromde/news2021/news.2021.de.btfromde.ppl13.en"
tgt_file = "en-de/bt_fromde/news2021/news.2021.de.btfromde.ppl13.de"
sco_file = "en-de/bt_fromde/news2021/news.2021.de.btfromde.ppl13.cometkiwi22.sco"
output_file = "en-de/bt_fromde/news2021/news.2021.de.btfromde.ppl13.prompted"

src_file = "en-de/testsets/newstest14-21-origlang-en.en"
tgt_file = "en-de/testsets/newstest14-21-origlang-en.de"
sco_file = "en-de/testsets/newstest14-21-origlang-en.cometkiwi22.sco"
output_file = "en-de/testsets/newstest14-21-origlang-en.prompted.TowerchatML"

src_file = "en-de/testsets/newstest2022-src.en"
tgt_file = "en-de/testsets/newstest2022-ref.de"
output_file = "en-de/testsets/newstest2022.prompted.TowerchatML"
"""
with open(src_file, "rb") as src, open(tgt_file, "rb") as tgt, open(sco_file, "rb") as sco, open(output_file, "w") as out_file:
    # Loop through each line in the input file
    for (srcline, tgtline, scoline) in zip(src, tgt, sco):
        if float(scoline.decode("utf-8")) > 0.0: # 0.79:
            srcline = srcline.decode("utf-8")
            tgtline = tgtline.decode("utf-8")
            # CAREFULL eos_token (<|im_end|> will be added by EOLE at training and inference)
            prompt = f"<|im_start|>user｟newline｠Translate the following text from English into German.｟newline｠English: {srcline.strip()}｟newline｠German: <|im_end|>｟newline｠<|im_start|>assistant｟newline｠{tgtline.strip()}"  # noqa: E501
            #prompt = f"<|im_start|>system｟newline｠<|im_end|>｟newline｠<|im_start|>user｟newline｠Translate the following text from English into German.｟newline｠English: {srcline.strip()}｟newline｠German: <|im_end|>｟newline｠<|im_start|>assistant｟newline｠{tgtline.strip()}<|im_end|>"  # noqa: E501
            #prompt = f"<|im_start|>system｟newline｠<|im_end|>｟newline｠<|im_start|>user｟newline｠Translate the following text from English into German.｟newline｠English: {srcline.strip()}｟newline｠German: <|im_end|>｟newline｠<|im_start|>assistant｟newline｠<|im_end|>"  # noqa: E501
            #prompt = f"<|start_header_id|> user<|end_header_id|>｟newline｠Translate the following text from English into German.｟newline｠English: {srcline.strip()}｟newline｠German: <|eot_id|><|start_header_id|> assistant<|end_header_id|>｟newline｠{tgtline.strip()}<|eot_id|>"  # noqa: E501
            #prompt = f"<|start_header_id|> system<|end_header_id|>You are a language identification system<|eot_id|><|start_header_id|> user<|end_header_id|>What language is: {srcline.strip()}<|eot_id|><|start_header_id|> assistant<|end_header_id|>English<|eot_id|>"  # noqa: E501
            out_file.write(prompt + "\n")
"""
src_file = "en-de/testsets/newstest2022-src.en"
tgt_file = "en-de/testsets/newstest2022-ref.de"
#output_file = "en-de/testsets/newstest2022-src-prompt.en"
output_file = "en-de/testsets/newstest2022.prompted.TowerchatML"

with open(src_file, "rb") as src, open(tgt_file, "rb") as tgt, open(output_file, "w") as out_file:
    # Loop through each line in the input file
    for (srcline, tgtline) in zip(src, tgt):
        srcline = srcline.decode("utf-8")
        tgtline = tgtline.decode("utf-8")
        #prompt = f"<|start_header_id|>user<|end_header_id|>｟newline｠Translate the following text from English into German.｟newline｠English: {srcline.strip()}｟newline｠German:<|eot_id|><|start_header_id|>assistant<|end_header_id|>｟newline｠"  # noqa: E501
        #prompt = f"<|im_start|>system｟newline｠<|im_end|>｟newline｠<|im_start|>user｟newline｠Translate the following text from English into German.｟newline｠English: {srcline.strip()}｟newline｠German:<|im_end|>｟newline｠<|im_start|>assistant｟newline｠"  # noqa: E501
        prompt = f"<|im_start|>user｟newline｠Translate the following text from English into German.｟newline｠English: {srcline.strip()}｟newline｠German: <|im_end|>｟newline｠<|im_start|>assistant｟newline｠{tgtline.strip()}"  # noqa: E501
        out_file.write(prompt + "\n")
"""
