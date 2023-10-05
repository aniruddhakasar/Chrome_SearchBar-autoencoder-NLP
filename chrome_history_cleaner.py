import pandas as pd
import numpy as np
import re
import os

## code to clean the chrome history
files = os.listdir('Data')
for i in files:
    if i.endswith('.csv'):
        file_path = os.path.join(f'Data/{i}')
        df = pd.read_csv(file_path)
        title = df[['title']]


        unique_titles = title.drop_duplicates()
        unique_titles.reset_index(inplace=True)
        unique_titles = unique_titles.drop('index',axis=1)
        unique_titles['title'] = unique_titles['title'].astype('str')

        # to get only all those history which has length > 2
        new_titles = []
        for i in range(len(unique_titles)): 
            if  2 < len(unique_titles['title'][i].split()):
                new_titles.append(unique_titles['title'][i])

        df2 = pd.DataFrame({'title':new_titles})

        # to remove all those linc history from my search history
        removed_lincs = []
        for i in range(len(df2)):
            if 0 == df2['title'][i].count('/'):
                removed_lincs.append(df2['title'][i])

        df3 = pd.DataFrame({'title':removed_lincs})


# to clean the search history text
        for i in range(len(df3)):
            lowered = df3['title'][i].lower()
            cleaned_content = re.sub("[^a-zA-Z]"," ",lowered)  
            df3['title'][i] = cleaned_content

        # to concatenate all the history as a paragraph
        text = ""
        for i in range(len(df3)):
            text += f"{df3['title'][i]}"

        # to remove out the extra space from the text
        tex2 = []
        for i in [''.join(i) for i in text.split(' ')]:
            if i == '':
                pass
            else:
                tex2.append(i)

        # to get final data as a paragraph
        final_text = ""
        for i in tex2:
            final_text += f" {i}"
        
        # to save the cleaned history files
        base_file  = os.path.basename(file_path)
        cleaned_file_name = base_file.replace('.csv','.txt')
        # to save final data as a textfile
        with open(f'Data/cleaned_History_files/{cleaned_file_name}','w') as file:
            file.write(final_text)
        print(f"{cleaned_file_name} is successfully in directory")
    else:
        pass
   