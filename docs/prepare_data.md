# Prepare Dataset

## LJSpeech

### Download Dataset

```
mkdir -p data/raw/ljspeech
cd data/raw
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
bzip2 -d LJSpeech-1.1.tar.bz2
tar -xvf LJSpeech-1.1.tar
cd ../../
```

### Forced Align and Preprocess Dataset

```
# Preprocess step: text and unify the file structure.
python data_gen/tts/runs/preprocess.py --config egs/lj/base_text2mel.yaml
# Align step: MFA alignment.
python data_gen/tts/runs/train_mfa_align.py --config egs/lj/base_text2mel.yaml
# Binarization step: Binarize data for fast IO. You only need to rerun this line when running different task if you have `preprocess`ed and `align`ed the dataset before.
python data_gen/tts/runs/binarize.py --config egs/lj/base_text2mel.yaml
```

## Biaobei

Please download the binarized dataset from [this link]()

## LibriTTS

### Download Dataset

```
mkdir -p data/raw/libritts
cd data/raw
wget https://www.openslr.org/resources/60/train-clean-100.tar.gz
wget https://www.openslr.org/resources/60/train-clean-360.tar.gz
# then unzip them into the data/raw/libritts folder and go back to the root dir.
```

### Forced Align and Preprocess Dataset

```
# Preprocess step: text and unify the file structure.
python data_gen/tts/runs/preprocess.py --config egs/libritts/base_text2mel.yaml
# Align step: MFA alignment.
python data_gen/tts/runs/train_mfa_align.py --config egs/libritts/base_text2mel.yaml
# Binarization step: Binarize data for fast IO. You only need to rerun this line when running different task if you have `preprocess`ed and `align`ed the dataset before.
python data_gen/tts/runs/binarize.py --config egs/libritts/base_text2mel.yaml
```

## More datasets will be supported soon...
