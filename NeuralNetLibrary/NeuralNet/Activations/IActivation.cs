using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetLibrary
{
    public interface IActivation
    {
        public double Activate(double d);
        public double Deactivate(double d);
        
        public Vector<double> Activate(Vector<double> vector);
        public Vector<double> Deactivate(Vector<double> vector);




    }
}