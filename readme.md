## The strcutre of the data:
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
>>>> frame0.png\
>>>> frame1.png
>>>> ...  

>>> 1
>>>> frame0.png\
>>>> ...

>>> ...  

>> speaker2\
>>> 0
>>>> frame0.png\
>>>> frame1.png
>>>> ...

>>> 1  
>>>> frame0.png \
>>>> ...

>>> ... 

>> speaker...
        
## notes:
any videos that don't start their name with a number will be ignored. \
any video that is longer than a 1 sec and 30 msec, the frames that exceeds the amount will be discraded.
