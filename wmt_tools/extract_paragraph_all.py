import xml.etree.ElementTree as ET

# Collect the processed lines


# Define the collection IDs to exclude
collections = ['general']  # Add the collection ID(s) you want to exclude

lgsrc = ['cs', 'de', 'en', 'fr', 'he', 'ja', 'liv', 'ru', 'sah', 'uk', 'zh']
lgtgt = ['cs', 'de', 'en', 'fr', 'he', 'ja', 'liv', 'ru', 'sah', 'uk', 'zh']
pairs = []
for src in lgsrc:
    for tgt in lgtgt:
        pairs.append(src + '-' + tgt)

for pair in pairs:

    output_lines = []
    # Load and parse the XML file
    input_file = 'wmttest2021/newstest2021.' + pair + '.all.xml'  # replace with your actual file path if needed
    try:
        print(input_file)
        tree = ET.parse(input_file)
        root = tree.getroot()

        # Process the docs within the collection if not excluded
        for doc in root.findall('.//doc'):
            # Skip <doc> elements that contain a <testsuite> child
            for src in doc.findall('.//src'):

                # For each <p> element within the <doc>, collect <seg> text
                parag = []
                for p in src.findall('.//p'):
                    # Collect the text from each <seg> element within <p>
                    segments = [seg.text for seg in p.findall('seg') if seg.text]
                    # Join each segment with '｟newline｠' and add to the output lines
                    parag.append('｟segsep｠'.join(segments))
                newdoc = '｟newline｠'.join(parag)
                if newdoc not in output_lines:
                    output_lines.append(newdoc)

        # Write the output to a text file
        with open('wmttest2021/docsep/newstest2021.src.' + pair + '.' + pair.split("-")[0], 'w', encoding='utf-8') as f:
            for line in output_lines:
                f.write(line + '\n')
    except:
        pass

