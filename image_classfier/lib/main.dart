import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_classfier/widgets/plant_recogniser.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const MainApp());
}

class MainApp extends StatelessWidget {
  const MainApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    SystemChrome.setPreferredOrientations(
      [
        DeviceOrientation.portraitUp,
        DeviceOrientation.portraitDown,
      ],
    );
    return MaterialApp(
      title: 'Face Shape',
      theme: ThemeData.light(),
      home: const PlantRecogniser(),
      debugShowCheckedModeBanner: false,
    );
  }
}