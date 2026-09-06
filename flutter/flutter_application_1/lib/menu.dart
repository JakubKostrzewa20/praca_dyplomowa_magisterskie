import 'package:flutter/material.dart';
import 'result.dart';
import 'package:image_picker/image_picker.dart' ;
// import 'package:tflite_flutter/tflite_flutter.dart';
// import 'package:image/image.dart'as img;


class MenuScreen extends StatelessWidget {
  const MenuScreen({super.key});


  
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
          child:Column(
          children:[ 
            const SizedBox(height:200),
            const SizedBox(height:200),
            const Text("Praca dyplomowa",
            style: TextStyle(
      fontSize: 30,
      color: Colors.white,
      fontWeight: FontWeight.bold,
    ),),
             const SizedBox(height:200),
            OutlinedButton(onPressed:() =>_modelOperation(context),
        style: OutlinedButton.styleFrom(
          backgroundColor:const Color.fromARGB(255, 118, 118, 118),
          foregroundColor:const Color.fromARGB(255, 63, 63, 63)),   
        child:const Text("Wybierz zdjęcie", style: TextStyle(
          color: Colors.white
        ),) 
        ),
        const SizedBox(height:100),
    Text("Jakub Kostrzewa",
    style: TextStyle(
      color: Colors.white,
      fontWeight: FontWeight.bold,
    ),)
        ] 
    ))));
  }
}
