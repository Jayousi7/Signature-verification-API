import requests

def verify_images():
    url = "http://127.0.0.1:8000/verify" 
    
    # insert your images path here 
    image1_path = r'.signatures\signatures_55\\forgeries_55_1.png'
    image2_path = r'.signatures\signatures_55\\original_55_1.png'

    with open(image1_path, 'rb') as f1, open(image2_path, 'rb') as f2:
        
        files = {
            'file1': (image1_path, f1, 'image/png'),
            'file2': (image2_path, f2, 'image/png')
        }

        data = {
            
        }

        
        print("Sending images to API...")
        response = requests.post(url=url, files=files, data=data)

    print(f'Status code: {response.status_code}')
    
    try:
        print(f'Response JSON: {response.json()}')
    except requests.exceptions.JSONDecodeError:
        print(f'Response Text (Error): {response.text}')
        
    print(f'Full response headers: {response.headers}')

if __name__ == "__main__":
    verify_images()