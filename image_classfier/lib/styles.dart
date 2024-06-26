import 'package:flutter/material.dart';

const kMainFont = 'Roboto';
const kButtonFont = 'Roboto';
const kDisplayFont = 'SquadaOne';

const kColorGreen = Color(0xFF395144);
const kColorLightGreen = Color(0XFF4E6C50);
const kColorBrown = Color(0XFFAA8B56);
const kColorLightYellow = Color(0xFFF0EBCE);

const kColorRed = Color(0xFFD96666);
const kColorLightRed = Color(0xFFF2CECE);
const kColorLightGray = Color(0xFFDDDDDD);

const kColorHunterGreen = Color(0xFF386641);
const kColorMayGreen = Color(0xFF6a994e);
const kColorAndroidGreen = Color(0xFFa7c957);
const kColorEggshell = Color(0xFFf2e8cf);
const kColorBitterSweetShimmer = Color(0xFFbc4749);

const kBgColor = kColorGreen;

const kTitleTextStyle = TextStyle(
  fontFamily: kDisplayFont,
  fontSize: 25.0,
  color: Colors.white,
  decoration: TextDecoration.none,
);

const kAnalyzingTextStyle = TextStyle(
    fontFamily: kMainFont,
    fontSize: 20.0,
    color: Colors.black,
    shadows: [
      Shadow(offset: Offset(0,0), color: Colors.white, blurRadius: 10)
    ],
    decoration: TextDecoration.none);

const kResultTextStyle = TextStyle(
    fontFamily: kDisplayFont,
    fontSize: 20.0,
    color: Colors.black,
    decoration: TextDecoration.none);

const kResultRatingTextStyle = TextStyle(
    fontFamily: kMainFont,
    fontSize: 15.0,
    color: Colors.black,
    decoration: TextDecoration.none);
