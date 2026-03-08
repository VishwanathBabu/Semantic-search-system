# step 1 : Build Docker image
docker build -t semantic-search-api .

# step 2 : Run the container 
docker run -p 8000:8000 semantic-search-api

# step 3 : open your index.html page in your browser


# Deployed website link
https://semantic-search-system-1jjg.onrender.com/