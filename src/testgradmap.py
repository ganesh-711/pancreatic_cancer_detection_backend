from gradcam import generate_heatmap

image_path = "../dataset/test/Cancer/1-008.jpg"
result = generate_heatmap(image_path)

print("Heatmap saved at:", result)
