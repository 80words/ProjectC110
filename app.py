# import the opencv library
import cv2
import tensorflow as tf
import numpy as np
model=tf.keras.models.load_model("converted_keras2/keras_model.h5")

  
# define a video capture object
vid = cv2.VideoCapture(0)
  
while(True):
      
    # Capture the video frame by frame
    ret, frame = vid.read()
    
    frame = cv2.flip(frame , 1)
    
    #resizing the image
    image=cv2.resize(frame, (224, 224), cv2.INTER_AREA)
    image=np.array(image, dtype=np.float32)
    image=np.expand_dims(image, axis=0)
    
    #normalizing the image
    normalized_image=image/255.0
    prediction = model.predict(normalized_image)
    print(prediction)
    
    # Display the resulting frame
    cv2.imshow('feed', frame)
      
    # Quit window with spacebar
    
    if cv2.waitKey(1) == 32:
        break
  
# After the loop release the cap object
vid.release()

# Destroy all the windows
cv2.destroyAllWindows()