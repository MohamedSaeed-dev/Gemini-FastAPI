from fastapi import FastAPI

# Create an instance of FastAPI
app = FastAPI()

# Define a simple route
@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI"}

# Define another route with a parameter
@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)