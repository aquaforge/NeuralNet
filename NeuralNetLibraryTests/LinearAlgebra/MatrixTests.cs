using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetLibrary;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace NeuralNetLibrary.Tests
{
    [TestClass()]
    public class MatrixTests
    {
        [TestMethod()]
        public void MatrixSerialize()
        {
            Random random = new();

            for (int i = 0; i < 100; i++)
            {
                Matrix? m1, m2;
                string s1, s2;

                m1 = new Matrix(random.Next(1, 15), random.Next(1, 10), random);
                s1 = JsonSerializer.Serialize(m1, new JsonSerializerOptions() { WriteIndented = true });

                m2 = JsonSerializer.Deserialize<Matrix>(s1);
                s2 = JsonSerializer.Serialize(m2, new JsonSerializerOptions() { WriteIndented = true });

                Assert.AreEqual(s1, s2);
                Assert.AreEqual(m1.SizeX, m2?.SizeX);
                Assert.AreEqual(m1.SizeY, m2?.SizeY);
            }
        }

    }
}