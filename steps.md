# installing and deploying ML program

pip install -r requirements.txt   
python train_and_save.py   
python app.py
curl -X POST \
  http://127.0.0.1:5000/predict \
  -H 'Content-Type: application/json' \
  -d '{"YearsExperience": 5.0}'
      
        
# Genrating the image
podman machine init
podman machine start
podman build -t salary-predictor-api . -f ContainerFile


# pushing image to github

podman login ghcr.io
podman tag salary-predictor-api:latest ghcr.io/abhishekmishra1069/salary-predictor-api:24112025
podman push ghcr.io/abhishekmishra1069/salary-predictor-api:24112025