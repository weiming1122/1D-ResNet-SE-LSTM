mkdir -p data/ISRUC_S1/ExtractedChannels
mkdir -p data/ISRUC_S1/RawData
echo 'Make data dir: data/ISRUC_S1'

cd data/ISRUC_S1/RawData

for i in `seq 1 100`
do
    wget http://dataset.isr.uc.pt/ISRUC_Sleep/subgroupI/$i.rar
    unrar x $i.rar
done

echo 'Download Data to "data/ISRUC_S1/RawData" complete.'

cd data/ISRUC_S1/ExtractedChannels

for i in `seq 1 100`
do
    wget http://dataset.isr.uc.pt/ISRUC_Sleep/ExtractedChannels/subgroupI-Extractedchannels/subject$i.mat
done
echo 'Download ExtractedChannels to "data/ISRUC_S1/ExtractedChannels" complete.'