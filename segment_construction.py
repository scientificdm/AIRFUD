# Segment construction from annotated documents in the txt format
#
import re
import sys

fileName = sys.argv[1] # 'PLU_Montpellier_ZONE-A_annotated', 'PPRI_Grabels_annotated', etc.
#%%
# Parse txt files with original text and split to pages:
f = open("Documents_txt/"+fileName+".txt", "r")

page = 0
lastPage = 0

pageBlocks = []
documentTitle = ''

pageText = ""
for line in f:
    try:
        if '>>>' in line:
            page = int(line[7:].strip())+1
            if page != 0:
                pageBlocks.append((pageText,lastPage))
                pageText = ""
        else:
            if documentTitle == '':
                documentTitle = line.strip()
            else:
                if pageText == '':
                    if line.strip() != '':
                        pageText+=line
                        lastPage = page
                else:
                    pageText+=line
                    lastPage = page
    except ValueError:
        print('Invalid input:',line)

# Save the last page:
pageBlocks.append((pageText,lastPage))

f.close()
#%%
# Parse text in pages and split them into blocks:

# Preliminary blocks:
candidateBlocks = []
newBlock = True

for page in pageBlocks:
    blockList = page[0].split('\n \n \n')
    
    for block in blockList:
        if block != '':
            candidateBlocks.append((block,page[1],newBlock))
            newBlock = True
            
    if blockList[-1] == '':
        newBlock = True
    else:
        newBlock = False

# Final blocks:
textBlocks = []
tmpBlock = ""
tmpPage = 0

for block in candidateBlocks:
    if block[2] == True:
        if tmpBlock != "":
            textBlocks.append((tmpBlock,tmpPage))
        tmpBlock = block[0]
        tmpPage = block[1]
    else:
        tmpBlock += block[0]

textBlocks.append((tmpBlock,tmpPage))
#%%
# Construct segments:
    
# Split large blocks into segments:
textSegments = []

for i in range(len(textBlocks)):
    # Split large blocks into segments (small blocks):
    segmTmp = re.sub(' \n\n','\n \n',textBlocks[i][0])
    segmTmp = re.sub(' \n  \n','\n \n',segmTmp)
    segmTmp = re.sub(' \n   \n','\n \n',segmTmp)
    blockList = segmTmp.split('\n \n')
    for block in blockList:
        if block != '':
            textSegments.append((block,i,textBlocks[i][1])) # text, N of large block, page of large block

# Construct multi-label segments:
segmentsLabeled = []
title = ''
subTitle = ''
subSubTitle = ''

for i in range(len(textSegments)):
    segmText = textSegments[i][0].strip()
    if '***' in segmText:
        title = segmText.split('***')[1].strip()
        subTitle = ''
        subSubTitle = ''
    elif '**' in segmText:
        subTitle = segmText.split('**')[1].strip()
        subSubTitle = ''
    else:
        # Check pertinance:
        verifiable = False
        nonverifiable = False
        informative = False
        if '^^' in segmText:
            verifiable = True
            segmText = segmText.split('^^')[1].strip()
        elif '<<' in segmText:
            nonverifiable = True
            segmText = segmText.split('<<')[1].strip()
        elif '>>' in segmText:
            informative = True
            segmText = segmText.split('>>')[1].strip()
        # Check sub sub title:
        if (subSubTitle == '') and ((segmText[-1] == ':' and segmText[1] != ')') or segmText[1] == ')'):
            subSubTitle = segmText
        elif (subSubTitle != '') and ((subSubTitle[-1] == ':' and subSubTitle[1] != ')' and segmText[-1] == ':' and segmText[1] != ')') or (subSubTitle[1] == ')' and  segmText[1] == ')')):
            subSubTitle = segmText
        else:
            if subSubTitle != '':
                if ((segmText[0] == '•') or (segmText[0] == '−') or (segmText[0] == '-')) or (subSubTitle[1] == ')') or (subSubTitle[0] == '•'):
                    if subTitle != '':
                        smallBlockText = title + '\n \n' + subTitle + '\n \n' + subSubTitle + '\n \n' + segmText
                    else:
                        smallBlockText = title + '\n \n' + subSubTitle + '\n \n' + segmText
                else:
                    subSubTitle = ''
                    if subTitle != '':
                        smallBlockText = title + '\n \n' + subTitle + '\n \n' + segmText
                    else:
                        smallBlockText = title + '\n \n' + segmText
            else:
                if subTitle != '':
                    smallBlockText = title + '\n \n' + subTitle + '\n \n' + segmText
                else:
                    smallBlockText = title + '\n \n' + segmText
            # Construct segment:
            if verifiable:
                segmentsLabeled.append((smallBlockText,textSegments[i][1],textSegments[i][2],documentTitle,'Verifiable')) # format: text, N of large block, N of page of large block, document title, label
            elif nonverifiable:
                segmentsLabeled.append((smallBlockText,textSegments[i][1],textSegments[i][2],documentTitle,'Non-verifiable')) # format: text, N of large block, N of page of large block, document title, label
            elif informative:
                segmentsLabeled.append((smallBlockText,textSegments[i][1],textSegments[i][2],documentTitle,'Soft')) # format: text, N of large block, N of page of large block, document title, label
            else:
                segmentsLabeled.append((smallBlockText,textSegments[i][1],textSegments[i][2],documentTitle,'False')) # format: text, N of large block, N of page of large block, document title, label
                
# Get some statistics:
print('Total segments:',len(segmentsLabeled))
print('Verifiable segments:',len([i for i in range(len(segmentsLabeled)) if segmentsLabeled[i][4]=='Verifiable']))
print('Non verifiable segments:',len([i for i in range(len(segmentsLabeled)) if segmentsLabeled[i][4]=='Non-verifiable']))
print('Soft segments:',len([i for i in range(len(segmentsLabeled)) if segmentsLabeled[i][4]=='Soft']))

# Save segments and their labels into a txt file:
f_out = open("Documents_txt/"+fileName+"_Segments.txt","w")

for i in range(len(segmentsLabeled)):
    f_out.write('>>> '+str(segmentsLabeled[i][4])+'\n\n')
    f_out.write(segmentsLabeled[i][0]+'\n\n\n')

f_out.close()
#%%