The input data is structered as follows:
> Videos\
>> speaker1\
>>> 0.mp4  
>>> 1.mp4  
>>> ...  

>> speaker2\  
>>> ...  

>> speaker...  

The preprocessed data is structered as follows:
> PreprocessedVideos\
>> speaker1\
>>> 0
>>>> frame0.png
>>>> frame1.png  
>>>> ...  

>>> 1  
>>>> frame0.png  
>>>> ...  

>>> ...  

>> speaker2\
>>> 0
>>>> frame0.png
>>>> frame1.png  
>>>> ...  

>>> 1  
>>>> frame0.png  
>>>> ...  

>>> ...   

>> speaker... 
        

any videos that dont start thier name with a number will be ignored
any videos that is longer than
