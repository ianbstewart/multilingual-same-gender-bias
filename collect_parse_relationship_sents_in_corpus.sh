# ES
#DATA_FILE=data/wiki/eswiki-20181001-corpus.xml.bz2
#LANG='es'
# FR
DATA_FILE=data/wiki/frwiki-20181001-corpus.xml.bz2
LANG='fr'
# IT
#DATA_FILE=data/wiki/itwiki-20181001-corpus.xml.bz2
#LANG='it'
python collect_parse_relationship_sents_in_corpus.py $DATA_FILE $LANG