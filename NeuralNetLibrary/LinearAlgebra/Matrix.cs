namespace NeuralNetLibrary
{
    [Serializable]
    public class Matrix : ICloneable, IEquatable<Matrix>
    {
        double[,] _matrix;
        public int SizeX => _matrix.GetLength(0);
        public int SizeY => _matrix.GetLength(1);


        public double[][] Data
        {
            get
            {

                var res = new double[SizeX][SizeY];
                for (int i = 0; i < SizeX; i++)
                {
                    var v = new double[SizeY];
                    for (int j = 0; j < SizeY; j++)
                        v[j] = _matrix[i, j];
                    res[i] = v;
                }
                return res;


                return _matrix;
            }
            //            set { _matrix = value; }
        }


        public Matrix(int x, int y, Random random) : this(x, y)
        {
            for (int i = 0; i < x; i++)
                for (int j = 0; j < y; j++)
                    _matrix[i, j] = random.NextDouble();
        }


        public Matrix() : this(0, 0) { }

        public Matrix(double[,] matrix)
        {
            _matrix = (double[,])matrix.Clone();
        }

        public Matrix(int x, int y)
        {
            _matrix = new double[x, y];
        }

        public double this[int x, int y]
        {
            get { return _matrix[x, y]; }
            set { _matrix[x, y] = value; }
        }

        public bool Equals(Matrix? other)
        {
            if (other == null && _matrix == null) return true;
            if (other == null || _matrix == null) return false;
            if (other.SizeX != SizeX || other.SizeY != SizeY) return false;
            for (int i = 0; i < SizeX; i++)
                for (int j = 0; j < SizeY; j++)
                    if (this[i, j] != other[i, j]) return false;
            return true;
        }

        public Matrix Copy() => new(_matrix);
        public object Clone() => Copy();


        public static Matrix operator -(Matrix m) => (-1) * m;

        public static Matrix operator *(double d, Matrix m) => m * d;
        public static Matrix operator *(Matrix m, double d)
        {
            if (m?._matrix == null) throw new ArgumentException("null matrix");

            var res = new double[m.SizeX, m.SizeY];

            for (int i = 0; i < m.SizeX; i++)
                for (int j = 0; j < m.SizeY; j++)
                    res[i, j] = m._matrix[i, j] * d;
            return new Matrix(res);
        }

        public static Vector operator *(Matrix m, Vector v)
        {
            if (m?._matrix == null) throw new ArgumentException("null matrix");
            if (v.Size != m.SizeY) throw new ArgumentException($"matrix[{m.SizeX},{m.SizeY}] and vector[{v.Size}] dimentions");

            var res = new double[m.SizeX];

            for (int i = 0; i < m.SizeX; i++)
            {
                double d = 0;

                for (int j = 0; j < m.SizeY; j++)
                    d += m._matrix[i, j] * v[j];
                res[i] = d;
            }
            return new Vector(res);
        }


        public static Matrix operator /(Matrix m, double d)
        {
            if (m?._matrix == null) throw new ArgumentException("null vector");

            var res = new double[m.SizeX, m.SizeY];

            for (int i = 0; i < m.SizeX; i++)
                for (int j = 0; j < m.SizeY; j++)
                    res[i, j] = m._matrix[i, j] / d;
            return new Matrix(res);
        }

        public Matrix Transpose()
        {
            if (_matrix == null) throw new ArgumentException("null matrix");

            var res = new double[SizeY, SizeX];

            for (int i = 0; i < SizeX; i++)
                for (int j = 0; j < SizeY; j++)
                    res[j, i] = _matrix[i, j];
            return new Matrix(res);
        }



    }
}