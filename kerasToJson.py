from tensorflow.keras.models import Sequential, load_model,model_from_json
from tensorflow.keras.utils import load_img,img_to_array

best_model = load_model('best.keras')
print (best_model.to_json())
