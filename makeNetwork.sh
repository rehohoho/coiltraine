NETWORKNAME=carlasimulator
DOCKERNAME=pytorch_whizz
sudo docker network prune
sudo docker network create --subnet=172.18.0.0/16 $NETWORKNAME
sudo docker network connect $NETWORKNAME $DOCKERNAME
