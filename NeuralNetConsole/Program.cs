// See https://aka.ms/new-console-template for more information
using NeuralNetLibrary;
using System.Text.Json;

Console.WriteLine("Hello, World!");

//Vector v = new Vector(new double[] { 3, 4 });
//string s = JsonSerializer.Serialize(v);
//Console.WriteLine(s);

Matrix matrix = new Matrix(5, 7, new Random());
string s = JsonSerializer.Serialize(matrix);
Console.WriteLine(s);

