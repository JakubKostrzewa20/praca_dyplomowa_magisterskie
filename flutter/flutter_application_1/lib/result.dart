import 'package:flutter/material.dart';
import 'dart:io';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';
import 'menu.dart';
import 'package:image_picker/image_picker.dart' ;
import 'dart:math' as math;

class ResultScreen extends StatefulWidget {
  final String imagePath;

  const ResultScreen({super.key, required this.imagePath});
  @override
  State<ResultScreen> createState() => _ResultScreenState();
}

class _ResultScreenState extends State<ResultScreen> {
  String className = 'Ładowanie...';
  String speciesName = 'Ładowanie...';
  String diseaseName = 'Ładowanie...';
  double probability = 0.0;

  late Interpreter interpreter;
  final List<String> labels = [ 'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy', ];
  @override
  void initState() {
    super.initState();
    _modelRun();
  }

  Future<void> _modelRun() async {
    interpreter = await Interpreter.fromAsset('lib/mobilenetv3small.tflite');
    final imageBytes = await File(widget.imagePath).readAsBytes();
    final img.Image? decodedImage = img.decodeImage(imageBytes);
    final rescaleImage = img.copyResize(decodedImage!, width: 224, height: 224);
    final imageMatrix = List.generate(
      rescaleImage.height,
      (y) => List.generate(rescaleImage.width, (x) {
        final pixel = rescaleImage.getPixel(x, y);
        //return [pixel.r, pixel.g, pixel.b];
        return [
  pixel.r / 255.0,
  pixel.g / 255.0,
  pixel.b / 255.0,
];
      }),
    );
    final input = [imageMatrix];
    final output = List<List<double>>.filled(1, List<double>.filled(38, 0));
    interpreter.allocateTensors();
    interpreter.run(input, output);
    
    
    final results = output.first;

    final maxLogit = results.reduce((a, b) => a > b ? a : b);

    final expValues = results
    .map((value) => math.exp(value - maxLogit))
    .toList();

final sumExp = expValues.reduce((a, b) => a + b);

final probabilities = expValues
    .map((value) => value / sumExp)
    .toList();

    int bestIndex = 0;
    double bestPropability = probabilities[0];
    for (int i = 1; i < probabilities.length; i++) {
      if (probabilities[i] > bestPropability) {
        bestPropability = results[i];
        bestIndex = i;
      }
    }
    final fullClassName = labels[bestIndex];

    print('OUTPUT MODELU: $results');
    print('Klasa: $bestIndex');
    print('Prawdopodobieństwo: $bestPropability');

    final parts = fullClassName.split('___');

    String species = parts[0];
    String disease = parts[1];

    setState(() {
      className = fullClassName; 
      speciesName = species; 
      diseaseName = disease; 
      probability = bestPropability;
    });
  }

  @override
  void dispose() {
    interpreter.close();
    super.dispose();
  }

  void _menuReturn(BuildContext context) {
    Navigator.push(
      context,
      MaterialPageRoute(builder: (_) => const MenuScreen()),
    );
  }
Future<void> _modelOperation(BuildContext context) async {
    final ImagePicker picker = ImagePicker();
    final XFile? pickedFile = await picker.pickImage(
      source: ImageSource.gallery,
    );

    if(pickedFile == null)
    {
      return;
    }

    final String imagePath = pickedFile.path;
    

    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_)=>  ResultScreen(
          imagePath: imagePath,
        )
        ) ,
      );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        color: const Color.fromARGB(255, 92, 75, 223),

        child: Center(
          child: Column(
            children: [
              const SizedBox(height: 100),
              Image.file(
                File(widget.imagePath),
                height: 300,
                fit: BoxFit.contain,
              ),
              const SizedBox(height: 100),
              Text(
                'Gatunek:$speciesName',
                style: TextStyle(
                  fontSize: 15,
                  color: Colors.white,
                  fontWeight: FontWeight.bold,
                ),
              ),
              const SizedBox(height: 20),
               Text(
                'Choroba:$diseaseName',
                style: TextStyle(
                  fontSize: 15,
                  color: Colors.white,
                  fontWeight: FontWeight.bold,
                ),
              ),
              const SizedBox(height: 20),
               Text(
                'Prawdopodobieństwo:${(probability * 100).toStringAsFixed(2)}%',
                style: TextStyle(
                  fontSize: 15,
                  color: Colors.white,
                  fontWeight: FontWeight.bold,
                ),
              ),
              const SizedBox(height: 40),
              OutlinedButton(
                onPressed: () => _modelOperation(context),
                style: OutlinedButton.styleFrom(
                  backgroundColor: const Color.fromARGB(255, 118, 118, 118),
                  foregroundColor: const Color.fromARGB(255, 63, 63, 63),
                ),
                child: const Text(
                  "Wybierz zdjęcie",
                  style: TextStyle(color: Colors.white),
                ),
              ),
              const SizedBox(height: 20),
              OutlinedButton(
                onPressed: () => _menuReturn(context),
                style: OutlinedButton.styleFrom(
                  backgroundColor: const Color.fromARGB(255, 118, 118, 118),
                  foregroundColor: const Color.fromARGB(255, 63, 63, 63),
                ),
                child: const Text(
                  "Powrót",
                  style: TextStyle(color: Colors.white),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
