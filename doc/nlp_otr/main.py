import doc.nlp_otr.redline_nlp_1 as rnlp
import pandas as pd
import numpy as np



def jsonfunc():
    
    truelist = rnlp.main()

    
    tops = rnlp.get_top_words(truelist)

    
    main_df = truelist.reset_index()
    main_df['top_features'] = tops



    main_df.columns = ['docID','content','top_features']
    

    # main_df.to_json('json-rows.json',orient='records')
    return main_df.iloc[np.random.randint(0,main_df.shape[0]-1),:].to_json(orient='records')
