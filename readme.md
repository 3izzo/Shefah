The input data is structered as follows:
Videos\
    speaker1\
        0 1.mp4
        4 2.mp4
        ...
    speaker2\
        ...
    speaker...

The preprocessed data is structered as follows:
PreprocessedVideos\
    speaker1\
        unmirrored
            0 1.mp4
            4 2.mp4
            ...
        mirrored
            0 1.mp4
            4 2.mp4
            ...
    speaker2\
        ...
    speaker...

any videos that dont start thier name with a number will be ignored
