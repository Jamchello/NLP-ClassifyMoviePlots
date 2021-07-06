#!/bin/bash
echo "Pulling updates..."
git pull

echo "Restarting server..."
pm2 restart all
