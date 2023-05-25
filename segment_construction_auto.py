# Automatic segment construction from non-annotated documents in the txt format
#
# For better quality of result it is strongly recommended to verify the text of the documents:
# each new tiltle should be separated by 2 empty lines (more precisely by: '\n \n \n'), 
# while other parts of the text s.t. subtitles, sub-subtitles, segment text should be  
# separated by 1 empty line (more precisely: '\n \n')
# 
import re
import sys

fileName = sys.argv[1] # 'PLU_Montpellier_ZONE-A', 'PPRI_Grabels', etc.
#%%
# PART I: Split original text into large labled blocks

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

# Prepare large blocks:
largeBlocksLabeled = []

for block in textBlocks:
    if block[0].strip() != '':
        largeBlocksLabeled.append((block[0].strip(),block[1],False)) # format: text, N of page, label
#%%
# PART II: Determine page numbers of small blocks (segments)

# Preliminary blocks:
candidateSegments = []
newSegment = True

for page in pageBlocks:
    segmTmp = re.sub(' \n\n','\n \n',page[0])
    segmTmp = re.sub(' \n  \n','\n \n',segmTmp)
    segmTmp = re.sub(' \n   \n','\n \n',segmTmp)
    blockList = segmTmp.split('\n \n')
    
    for block in blockList:
        if block != '':
            candidateSegments.append((block,page[1],newSegment))
            newSegment = True
            
    if blockList[-1] == '':
        newSegment = True
    else:
        newSegment = False

# Final blocks:
textSegments = []
tmpSegment = ""
tmpPage = 0

for segment in candidateSegments:
    if segment[2] == True:
        if tmpSegment != "":
            textSegments.append((tmpSegment.strip(),tmpPage))
        tmpSegment = segment[0]
        tmpPage = segment[1]
    else:
        tmpSegment += segment[0]

textSegments.append((tmpSegment.strip(),tmpPage))
#%
# PART III: Split large labled blocks into small labled blocks (segments)

# Construct segments:
segmentsLabeled = []

k = 0
title = ''
subTitle = ''
subSubTitle = ''
titles = []
for i in range(len(largeBlocksLabeled)):
    block = largeBlocksLabeled[i]
    blockTmp = re.sub(' \n\n','\n \n',block[0])
    blockTmp = re.sub(' \n  \n','\n \n',blockTmp)
    blockTmp = re.sub(' \n   \n','\n \n',blockTmp)
    segments = blockTmp.split('\n \n')
    segmentsFlag = False
    emptySubTitle = False
    emptySubSubTitle = False
    for j in range(len(segments)):
        if segments[j].strip() != '':
            if j == 0:
                title = segments[0]
                titles.append(title.strip())
                subTitle = ''
                subSubTitle = ''
            elif j > 0:
                if subTitle == '':
                    if segments[j].isupper():
                        subTitleType = 0
                        subTitle = segments[j]
                        titles.append(subTitle.strip())
                        subSubTitle = ''
                        emptySubTitle = True
                    elif segments[j][0].isdigit() and segments[j][1]==')':
                        subTitleType = 1
                        subTitle = segments[j]
                        titles.append(subTitle.strip())
                        subSubTitle = ''
                        emptySubTitle = True
                    elif segments[j].strip()[-1] == ':' and not (segments[j][0].isdigit() and segments[j][1]==')'):
                        subTitleType = 2
                        subTitle = segments[j]
                        titles.append(subTitle.strip())
                        subSubTitle = ''
                        emptySubTitle = True
                    else:
                        print('Error! Subtitle is not detected in',title)
                elif not emptySubTitle and not emptySubSubTitle and (subTitleType == 0) and segments[j].isupper():
                    subTitle = segments[j]
                    titles.append(subTitle.strip())
                    subSubTitle = ''
                    emptySubTitle = True
                elif not emptySubTitle and not emptySubSubTitle and (subTitleType == 1) and segments[j][0].isdigit() and segments[j][1]==')':
                    subTitle = segments[j]
                    titles.append(subTitle.strip())
                    subSubTitle = ''
                    emptySubTitle = True
                elif not emptySubTitle and not emptySubSubTitle and (subTitleType == 2) and segments[j].strip()[-1] == ':' and not segments[j][1]==')' and not segments[j].strip()[0]=='•' and ("Définition" in segments[j] or "Dans l'ensemble" in segments[j]):
                    subTitle = segments[j]
                    titles.append(subTitle.strip())
                    subSubTitle = ''
                    emptySubTitle = True
                elif not emptySubSubTitle and ((segments[j].strip()[-1] == ':') or ((segments[j][0] == 'a' or segments[j][0] == 'b' or segments[j][0] == 'c' or segments[j][0] == 'd') and segments[j].strip()[1] == ')')):
                    subSubTitle = segments[j]
                    emptySubSubTitle = True
                    titles.append(subSubTitle.strip())
                else:
                    if subSubTitle != '':
                        if ((segments[j][0] == '•') or (segments[j][0] == '−') or (segments[j][0] == '-')) or (subSubTitle.strip()[1] == ')') or (subSubTitle.strip()[0] == '•'):
                            segmentText = title + '\n \n' + subTitle + '\n \n' + subSubTitle + '\n \n' + segments[j].strip()
                        else:
                            subSubTitle = ''
                            segmentText = title + '\n \n' + subTitle + '\n \n' + segments[j].strip()
                    else:
                        segmentText = title + '\n \n' + subTitle + '\n \n' + segments[j].strip()
                    if block[2] == False:
                        segmentsLabeled.append((segmentText,i,block[1],documentTitle,False)) # format: text, N of block, N of page, document title
                    segmentsFlag = True
            k+=1
    if not segmentsFlag:
        if subTitle != '':
            if subSubTitle != '':
                segmentText = title + '\n \n' + subTitle + '\n \n' + subSubTitle + '\n \n' + segments[j].strip()
            else:
                segmentText = title + '\n \n' + subTitle + '\n \n' + segments[j].strip()
        else:
            segmentText = title + '\n \n' + segments[j].strip()
        segmentsLabeled.append((segmentText,i,block[1],documentTitle)) # format: text, N of block, N of page, document title
                
# Get some statistics:
print('Total segments:',len(segmentsLabeled))

# Save segments and their labels into a txt file:
f_out = open("Documents_txt/"+fileName+"_Segments.txt","w")

for i in range(len(segmentsLabeled)):
    f_out.write('>>> \n\n')
    f_out.write(segmentsLabeled[i][0]+'\n\n\n')

f_out.close()
#%%