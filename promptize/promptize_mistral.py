# Input and output file paths
"""
input_file_path = "en-de/testsets/newstest2014-origlang-en.en"
output_file_path = "en-de/testsets/newstest2014-origlang-en.prompt.eurollmchat.en"

input_file_path = "en-de/testsets/newstest2022-src.en"
output_file_path = "en-de/testsets/newstest2022-src.prompt.en"
input_file_path = "en-de/testsets/newstest2023-src.en"
output_file_path = "en-de/testsets/newstest2023-src.prompt.eurollmchat.en"

input_file_path = "en-de/testsets/wmt2024-src.en"
output_file_path = "en-de/testsets/wmt2024-src.prompt.3shot.en"
input_file_path = "en-de/testsets/wmt2024-src.en"
output_file_path = "en-de/testsets/wmt2024-src.prompt.mistral-3shot.en"

input_file_path = "en-de/testsets/wmttest2024.literary.src.en-de.en.en"
output_file_path = "en-de/testsets/wmt2024-literary-src.prompt.eurollmchat.en"

input_file_path = "/home/vincent/Downloads/wmttest2024.src.en-xx.txt"
output_file_path = "en-de/testsets/wmttest2024.src.en-xx.llamachat.3shot.en"
input_file_path = "/mnt/InternalCrucial4/LLM_work/dialogsum-long.en"
output_file_path = "/mnt/InternalCrucial4/LLM_work/dialogsum-long.MLchat.en"
input_file_path = "/mnt/InternalCrucial4/LLM_work/prosocial-dialog-short.en"
output_file_path = "/mnt/InternalCrucial4/LLM_work/prosocial-dialog-short.MLchat.en"
input_file_path = "/mnt/InternalCrucial4/LLM_work/fineweb-literature-400k.en"
output_file_path = "/mnt/InternalCrucial4/LLM_work/fineweb-literature-400k.MLchat.en"
"""

input_file_path = "/mnt/InternalCrucial4/data/wmt/WMT24_GeneralMT/wmttest2024.src.en-de.en"
output_file_path = "/mnt/InternalCrucial4/data/wmt/WMT24_GeneralMT/wmttest2024.src.en-de.mistralchat.en"
#output_file_path = "/mnt/InternalCrucial4/data/wmt/WMT24_GeneralMT/wmttest2024.src.en-de.3shotllamachat.en"

# Open the input file for reading and the output file for writing
with open(input_file_path, "r") as input_file, open(
    output_file_path, "w"
) as output_file:
    # Loop through each line in the input file
    for line in input_file:
 
        #Mistral instruct 0 shot
        prompt = f"[INST]Translate the following text from English into German. Pay attention to the domain of the text and its terminology. Give only the translation and no explanation.｟newline｠English: {line.strip()}｟newline｠German: [/INST]"  # noqa: E501

        #Mistral instruct 3 shots
        #prompt = f"[INST]Translate the following text from English into German.｟newline｠English: White nationalists, gun fetishists, pro-dictator right wing.. there's the danger. Deport the white supremacists.｟newline｠German: [/INST]Weiße Nationalisten, Waffenfetischisten, Diktator-Freundliche vom rechten Flügel … da liegt die Gefahr. Deportiert die weißen Rassisten.</s>[INST]Translate the following text from English into German.｟newline｠English: It's 2023 and I still see people with iPhones actively avoid using Apple Maps.｟newline｠German: [/INST]Es ist 2023 und ich sehe immer noch Menschen mit iPhones, die aktiv die Verwendung von Apple Maps vermeiden.</s>[INST]Translate the following text from English into German.｟newline｠English: #CNN needs to just close down. Yesterday's fiasco with #MangoMoron was an outrageous fiasco. Who at that pathetic, dying network was responsible for vetting the so-called \"independent voters\" in the audience? Which cable news \"executive\" made the call to give the fat, lying criminal this much air-time?｟newline｠German: [/INST]#CNN muss einfach den Laden dichtmachen. Das Fiasko von gestern mit #MangoMoron war einfach ungeheuerlich. Wer war bei diesem jämmerlichen, im Sterben liegenden Sender eigentlich für die Überprüfung der sogenannten „unabhängigen Wähler“ im Publikum verantwortlich? Welcher Kabelnachrichten-„Verantwortliche“ hat denn die Entscheidung getroffen, dem fetten, lügenden Kriminellen so viel Sendezeit zu geben?</s>[INST]Translate the following text from English into German.｟newline｠English: {line.strip()}｟newline｠German: [/INST]"  # noqa: E501

        #Mistral 3 shots variant
        #prompt = f"[INST]Translate the following text from English into German. Give only one translation and no explanation.｟newline｠English: White nationalists, gun fetishists, pro-dictator right wing.. there's the danger. Deport the white supremacists.｟newline｠German: [/INST]Weiße Nationalisten, Waffenfetischisten, Diktator-Freundliche vom rechten Flügel … da liegt die Gefahr. Deportiert die weißen Rassisten.</s>[INST]Translate the following text from English into German. Give only one translation and no explanation.｟newline｠English: It's 2023 and I still see people with iPhones actively avoid using Apple Maps.｟newline｠German: [/INST]Es ist 2023 und ich sehe immer noch Menschen mit iPhones, die aktiv die Verwendung von Apple Maps vermeiden.</s>[INST]Translate the following text from English into German. Give only one translation and no explanation.｟newline｠English: #CNN needs to just close down. Yesterday's fiasco with #MangoMoron was an outrageous fiasco. Who at that pathetic, dying network was responsible for vetting the so-called \"independent voters\" in the audience? Which cable news \"executive\" made the call to give the fat, lying criminal this much air-time?｟newline｠German: [/INST]#CNN muss einfach den Laden dichtmachen. Das Fiasko von gestern mit #MangoMoron war einfach ungeheuerlich. Wer war bei diesem jämmerlichen, im Sterben liegenden Sender eigentlich für die Überprüfung der sogenannten „unabhängigen Wähler“ im Publikum verantwortlich? Welcher Kabelnachrichten-„Verantwortliche“ hat denn die Entscheidung getroffen, dem fetten, lügenden Kriminellen so viel Sendezeit zu geben?</s>[INST]Translate the following text from English into German. Give only one translation and no explanation.｟newline｠English: {line.strip()}｟newline｠German: [/INST]"  # noqa: E501
        output_file.write(prompt + "\n")
