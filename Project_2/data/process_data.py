# Import Libraries
import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Data loading and combine datasets
    
    :params: messages_filepath - disaster_messages.csv dir path
             categories_filepath: disaster_categories.csv dir path
    
    :outputs: DataFrame
    
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how ='outer', on =['id'])
    return df



def clean_data(df):
    """
    Data Quality check
    
    :params: df - DataFrame
    :outputs: df - Cleaned DataFrame
    
    """
    # Divide the original DataFrame's category values into a new DataFrame.
    cate_cols = df['categories'].str.split(';', expand=True)
    row = cate_cols.head(1)
    
    # The lambda function will cut each string's characters down to the final two characters.
    cate_colnames = row.applymap(lambda x: x[:-2]).iloc[0,:]
    cate_cols.columns = cate_colnames
    
    # Iterate while retaining the string's final character.
    for col in cate_cols:
        cate_cols[col] = cate_cols[col].astype(str).str[-1]
        cate_cols[col] = cate_cols[col].astype(int)
    cate_cols['related'] = cate_cols['related'].replace(to_replace=2, value=1)
        
    # Drop original categories column
    df.drop('categories', axis=1, inplace=True)
    
    df = pd.concat([df, cate_cols], axis=1)
    df.drop_duplicates(inplace=True)
    return df


def save_data(df, database_filename):
    """This function will save df in a SQLite."""
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('disaster_messages_tb', engine, index=False, if_exists='replace')  
    

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()