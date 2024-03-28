import 'dart:io';
import 'package:flutter/material.dart';

import '../styles.dart';

class PlantPhotoView extends StatelessWidget {
  final File? file;
  const PlantPhotoView({super.key, this.file});

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 300,
      height: 300,
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(9),
        boxShadow: const [
          BoxShadow(
            blurRadius: 10,
            color: Colors.black54,
            offset: Offset(0,0),
            spreadRadius: -5
          )
        ]
      ),
      child: (file == null)
          ? _buildEmptyView()
          : Image.file(file!, fit: BoxFit.cover),
    );
  }

  Widget _buildEmptyView() {
    return const Center(
        child: Text(
      'Please pick a photo',
      style: kAnalyzingTextStyle,
    ));
  }
}
