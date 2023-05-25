# Extract text from urban planning documents in pdf
#
import fitz
import sys

fileName = sys.argv[1] # 'PLU_Montpellier_ZONE-A', 'PPRI_Grabels', etc.
#%%
def getText(document,offset):  
    text = ""
    # Join blocks from all pages:
    for i in range(len(document)):
        # Open the page:
        page = document[i].get_text()
        
        # Remove headers and footprints from pages:
        text += '>>> p. '+str(i)+'\n'
        if i==0:
            text += fileName+' \n ' # add name of the document
        text += page[offset:]
    
    return text

def getTextPPRI(document):
    text = ""
    
    # Join blocks from all pages:
    for i in range(len(document)):
        # Open the page:
        page = document[i].get_text()
        
        # Remove headers and footprints from pages:
        if i == 0:
            text += '>>> p. 0\n'
            text += fileName+' \n ' # add name of the document
            text += page[0:].lstrip()
        else:
            text += '>>> p. '+str(i)+'\n'
            text += page[8:].lstrip()
    
    return text

def getTextPPRIGrabels(document):
    text = ""
    
    # Join blocks from all pages:
    for i in range(len(document)):
        # Open the page:
        page = document[i].get_text()
        
        # Remove headers and footprints from pages:
        if i == 0:
            text += '>>> p. 0\n'
            text += fileName+' \n ' # add name of the document
            text += page[0:].lstrip()
        else:
            text += '>>> p. '+str(i)+'\n'
            text += page[:-7].lstrip()
    
    return text

def getTextPLUAll(document):
    text = ""
    
    # Join blocks from all pages:
    for i in range(len(document)):
        # Open the page:
        page = document[i].get_text()
        
        # Remove headers and footprints from pages:
        if i == 0:
            text += '>>> p. 0\n'
            text += fileName+' \n ' # add name of the document
            text += page[0:].lstrip()
        else:
            text += '>>> p. '+str(i)+'\n'
            text += page.split("IC’se")[1][7:].lstrip()
    
    return text

def getTextSRCE(document):
    text = ""
    
    # Join blocks from all pages:
    for i in range(len(document)):
        # Open the page:
        page = document[i].get_text()
        
        # Remove headers and footprints from pages:
        if i == 0:
            text += '>>> p. 0\n'
            text += fileName+' \n ' # add name of the document
            text += page[0:].lstrip()
        elif i == 9:
            text += '>>> p. '+str(i)+'\n'
            text += page[87:].lstrip()
        else:
            text += '>>> p. '+str(i)+'\n'
            text += page[88:].lstrip()
    
    return text

doc = fitz.open("Documents_PDF/"+fileName+".pdf")

# Extract text from all pages and remove headers:
if 'PPRI' in fileName:
    if 'Grabels' in fileName:
        docText = getTextPPRIGrabels(doc)
    else:
        docText = getTextPPRI(doc)
elif ('PLU' in fileName) and ('All' in fileName):
    docText = getTextPLUAll(doc)
elif 'SRCE' in fileName:
    docText = getTextSRCE(doc)
else:
    if ('ZONE-A-' in fileName) or ('ZONE-N' in fileName):
        docText = getText(doc,38)
    elif ('ZONE-AU0' in fileName) or ('ZONE-14AU' in fileName):
        docText = getText(doc,45)
    else:
        docText = getText(doc,46)
        
# Delete more than 2 empty lines:
docTextClean = ""
emptyLines = 0
for line in docText.splitlines():
    if line.strip() != "":
        docTextClean += line+'\n'
        emptyLines = 0
    elif emptyLines < 2:
        docTextClean += line+'\n'
        emptyLines += 1
        
# Save text as txt file:
f_out = open("Documents_txt/"+fileName+".txt","w")

f_out.write(docTextClean)

f_out.close()

# Report success of the operation:
print("File Documents_txt/"+fileName+".txt has been created!")
#%%