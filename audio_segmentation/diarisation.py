from pyannote.audio import Pipeline

path = 'D:/Big Data/test.wav'

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")#,
                                    #use_auth_token="ACCESS_TOKEN_GOES_HERE")

# 4. apply pretrained pipeline
diarization = pipeline(path)

# 5. print the result
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")