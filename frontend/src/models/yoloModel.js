// YOLOv8 Model Loader (Placeholder for future ONNX implementation)
// Currently using COCO-SSD for hackathon reliability

export const loadYoloModel = async () => {
  console.log('YOLO model placeholder - using COCO-SSD instead')
  return null
}

export const runYoloInference = async (model, imageData) => {
  // Placeholder for YOLO inference
  return []
}

export default {
  loadYoloModel,
  runYoloInference
}
