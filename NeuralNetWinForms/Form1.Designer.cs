namespace NeuralNetWinForms
{
    partial class Form1
    {
        /// <summary>
        ///  Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        ///  Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        ///  Required method for Designer support - do not modify
        ///  the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.components = new System.ComponentModel.Container();
            this.sharpPictureBox1 = new NeuralNetWinForms.SharpPictureBox();
            this.timer1 = new System.Windows.Forms.Timer(this.components);
            ((System.ComponentModel.ISupportInitialize)(this.sharpPictureBox1)).BeginInit();
            this.SuspendLayout();
            // 
            // sharpPictureBox1
            // 
            this.sharpPictureBox1.Dock = System.Windows.Forms.DockStyle.Fill;
            this.sharpPictureBox1.Location = new System.Drawing.Point(0, 0);
            this.sharpPictureBox1.Name = "sharpPictureBox1";
            this.sharpPictureBox1.Size = new System.Drawing.Size(800, 450);
            this.sharpPictureBox1.TabIndex = 0;
            this.sharpPictureBox1.TabStop = false;
            // 
            // timer1
            // 
            this.timer1.Enabled = true;
            this.timer1.Tick += new System.EventHandler(this.timer1_Tick);
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(7F, 15F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(800, 450);
            this.Controls.Add(this.sharpPictureBox1);
            this.Name = "Form1";
            this.Text = "Form1";
            ((System.ComponentModel.ISupportInitialize)(this.sharpPictureBox1)).EndInit();
            this.ResumeLayout(false);

        }

        #endregion

        private SharpPictureBox sharpPictureBox1;
        private System.Windows.Forms.Timer timer1;
    }
}