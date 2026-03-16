#!/bin/bash
# Script de déploiement pour la VM de production

VM_IP=$1
VM_USER=${2:-ubuntu}

if [ -z "$VM_IP" ]; then
    echo "Usage: ./deploy_vm.sh <VM_IP> [VM_USER]"
    exit 1
fi

echo "Déploiement sur la VM $VM_USER@$VM_IP..."

# On suppose que le projet est déjà cloné sur la VM dans ~/MLOPS
# Et que les clés SSH sont configurées
ssh $VM_USER@$VM_IP << EOF
    cd ~/MLOPS
    git pull origin main
    docker compose up --build -d
    docker system prune -f
EOF

echo "Déploiement terminé !"
