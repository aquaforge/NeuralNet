namespace NeuralNetLibrary
{
    [Serializable]
    public class Vector : ICloneable, IEquatable<Vector>
    {
        double[] _vector;
        public int Size => _vector.Length;



        public double[] Data
        {
            get { return _vector; }
            set { _vector = value; }
        }


        public Vector() : this(0) { }

        public Vector(double[] vector)
        {
            _vector = (double[])vector.Clone();
        }


        public Vector(int size, Random random) : this(size)
        {
            for (int i = 0; i < size; i++)
                _vector[i] = random.NextDouble();
        }

        public Vector(int size)
        {
            _vector = new double[size];
        }

        public double this[int pos]
        {
            get { return _vector[pos]; }
            set { _vector[pos] = value; }
        }


        public bool Equals(Vector? other)
        {
            if (other == null && _vector == null) return true;
            if (other == null || _vector == null) return false;
            if (other.Size != Size) return false;
            for (int i = 0; i < Size; i++)
                if (this[i] != other[i]) return false;
            return true;
        }

        public Vector Copy() => new(_vector);
        public object Clone() => Copy();

        static void CheckVectorsForBinaryOperation(Vector v1, Vector v2)
        {
            if (v1?._vector == null || v2?._vector == null) throw new ArgumentException("null vector");
            if (v1.Size != v2.Size) throw new ArgumentException($"wrong lenght {v1.Size} {v2.Size}");
        }

        public static Vector operator +(Vector v1, Vector v2)
        {
            CheckVectorsForBinaryOperation(v1, v2);

            var res = new double[v1.Size];

            for (int i = 0; i < v1.Size; i++)
                res[i] = v1._vector[i] + v1._vector[i];
            return new Vector(res);
        }
        public static Vector operator -(Vector v1, Vector v2)
        {
            CheckVectorsForBinaryOperation(v1, v2);

            var res = new double[v1.Size];

            for (int i = 0; i < v1.Size; i++)
                res[i] = v1._vector[i] - v1._vector[i];
            return new Vector(res);
        }


        public static Vector operator -(Vector v) => (-1) * v;



        public double Length => Math.Sqrt(_vector.Select(n => n * n).Sum());

        public Vector Normalize() => this / this.Length;



        public static Vector operator *(double d, Vector v1) => v1 * d;
        public static Vector operator *(Vector v1, double d)
        {
            if (v1?._vector == null) throw new ArgumentException("null vector");

            var res = new double[v1.Size];

            for (int i = 0; i < v1.Size; i++)
                res[i] = v1._vector[i] * d;
            return new Vector(res);
        }

        public static Vector operator /(Vector v1, double d)
        {
            if (v1?._vector == null) throw new ArgumentException("null vector");

            var res = new double[v1.Size];

            for (int i = 0; i < v1.Size; i++)
                res[i] = v1._vector[i] / d;
            return new Vector(res);
        }




    }
}