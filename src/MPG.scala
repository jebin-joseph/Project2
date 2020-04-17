import scalation.analytics.RegTechnique.{QR, RegTechnique}
import scalation.analytics.{ActivationFun, ELM_3L1, NeuralNet_3L, NeuralNet_XL, Optimizer_SGDM, Perceptron, RegTechnique, Regression, TranRegression, pullResponse}
import scalation.math.sq
import scalation.columnar_db.Relation
import scalation.linalgebra.{MatrixD, VectorD}
import scalation.util.banner
import scalation.plot.PlotM

import scalation.analytics.Optimizer._
//import scalation.analytics.Optimizer_SGD._
import scalation.analytics.Optimizer_SGDM._

import scala.math.{exp, log, sqrt}
import ActivationFun._

object MPG extends App {
    // Choose desired techniques by setting value of desired model to true
    val tran = false
    val perceptron = true
    val NN_3L = true
    val NN_XL = true
    val ELM = false

    // Choose desired tran functions
    val logarithm = true
    val square_root = true
    val square = true
    val expon = true

    // Choose desired activation functions
    val sigmoid_ = true
    val reLU_ = true
    val tanh_ = true
    val id_ = true

    val x_cols = Seq(0, 1, 2, 3, 4)
    val y_cols = Seq(5)
    val all_cols = Seq(0,1,2,3,4,5)
    val x = Relation("MPG_Full.csv", "MPG_Full_x", null, -1).toMatriD(x_cols)
    val y = Relation("MPG_Full.csv", "MPG_Full_Y", null, -1).toVectorD(5)
    val y_mat = Relation("MPG_Full.csv", "MPG_Full_Y_Mat", null, -1).toMatriD(y_cols)
    val xy = Relation("MPG_Full.csv", "MPG_Full_x_Y", null, -1).toMatriD(all_cols)
    val _1 = VectorD.one (x.dim1)
    val oxy = _1 +^: xy

    if (tran) {
        // log tran function and exp itran function
        if(logarithm) {
            val model_tran_0 = new TranRegression(x, y, null, null, log _, exp _, QR)
            val (cols, rSq) = model_tran_0.forwardSelAll(0, true)
            println (s"rSq = $rSq")
            val k = cols.size
            val t = VectorD.range (1, k)
            new PlotM (t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"), "R^2 vs n for log tran", lines = true)
            model_tran_0.analyze()
            println(model_tran_0.report)
            println(model_tran_0.summary)
        }

        //sqrt tran function and sq itran function
        if(square_root) {
            val model_tran_1 = new TranRegression(x, y, null, null, sqrt _, sq _, QR)
            val (cols, rSq) = model_tran_1.forwardSelAll(0, true)
            println (s"rSq = $rSq")
            val k = cols.size
            val t = VectorD.range (1, k)
            new PlotM (t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"), "R^2 vs n for sqrt tran", lines = true)
            model_tran_1.analyze()
            println(model_tran_1.report)
            println(model_tran_1.summary)
        }

        //sq tran function and sqrt itran function
        if(square) {
            print("\nResponse grows like a square root. ")
            val model_tran_2 = new TranRegression(x, y, null, null, sq _, sqrt _, QR)
            val (cols, rSq) = model_tran_2.forwardSelAll(0, true)
            println (s"rSq = $rSq")
            val k = cols.size
            val t = VectorD.range (1, k)
            new PlotM (t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"), "R^2 vs n for sq tran", lines = true)
            model_tran_2.analyze()
            println(model_tran_2.report)
            println(model_tran_2.summary)
        }

        //exp tran function and log itran function
        if(expon) {
            val model_tran_3 = new TranRegression(x, y, null, null, exp _, log _, QR)
            val (cols, rSq) = model_tran_3.forwardSelAll(0, true)
            println (s"rSq = $rSq")
            val k = cols.size
            val t = VectorD.range (1, k)
            new PlotM (t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"), "R^2 vs n for expon tran", lines = true)
            model_tran_3.analyze()
            println(model_tran_3.report)
            println(model_tran_3.summary)
        }
    }

    if (perceptron) {
        //sigmoid activation function
        if(sigmoid_) {
            val model_perc_0 = Perceptron(oxy)
            val (cols, rSq) = model_perc_0.forwardSelAll(0, true)
            println (s"rSq = $rSq")
            val k = cols.size
            val t = VectorD.range (1, k)
            new PlotM (t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"), "R^2 vs n for sigmoid perc", lines = true)
            model_perc_0.train().eval()
            println(model_perc_0.report)
          println(model_perc_0.summary)
        }
        //reLU activation function
        if(reLU_) {
            val model_perc_1 = Perceptron(oxy, f0 = f_reLU)
            model_perc_1.forwardSelAll(0,true)
//            no graph shown because of NaN values
            model_perc_1.train().eval()
            println(model_perc_1.report)
          println(model_perc_1.summary)
        }
        //tanh activation function
        if(tanh_) {
            val model_perc_2 = Perceptron(oxy, f0 = f_tanh)
            val (cols, rSq) = model_perc_2.forwardSelAll(0,true)
            println (s"rSq = $rSq")
            val k = cols.size
            val t = VectorD.range (1, k)
            new PlotM (t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"), "R^2 vs n for tanh perc", lines = true)
            model_perc_2.train().eval()
            println(model_perc_2.report)
          println(model_perc_2.summary)
        }
        //id activation function
        if(id_) {
            val model_perc_3 = Perceptron(oxy, f0 = f_id)
            model_perc_3.forwardSelAll(0,true)
//            no graph shown because of NaN values
            model_perc_3.train().eval()
            println(model_perc_3.report)
          println(model_perc_3.summary)
        }
    }

    if (NN_3L) {
        //sigmoid activation function
        if(sigmoid_) {
            val model_perc_0_0 = NeuralNet_3L(oxy, f1 = f_reLU)
            val (cols, rSq) = model_perc_0_0.forwardSelAll(0)
            println (s"rSq = $rSq")
            val k = cols.size
            val t = VectorD.range (1, k)
            new PlotM (t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"), "R^2 vs n for sigmoid tanh nn3l", lines = true)
            model_perc_0_0.train().eval()
            println(model_perc_0_0.report)

            val model_perc_0_1 = NeuralNet_3L(oxy, f1 = f_tanh)
            val (cols1, rSq1) = model_perc_0_1.forwardSelAll(0)
            println (s"rSq = $rSq1")
            val k1 = cols1.size
            val t1 = VectorD.range (1, k1)
            new PlotM (t1, rSq1.t, Array ("R^2", "R^2 bar", "R^2 cv"), "R^2 vs n for sigmoid tanh nn3l", lines = true)
            model_perc_0_1.train().eval()
            println(model_perc_0_1.report)
        }
        //reLU activation function
        if(reLU_) {
            val model_perc_1_0 = NeuralNet_3L(oxy, f0 = f_reLU)
            model_perc_1_0.forwardSelAll(0)
//          no graph shown because of NaN values
            model_perc_1_0.train().eval()
            println(model_perc_1_0.report)
        }
        //tanh activation function
        if(tanh_) {
            val model_perc_2_0 = NeuralNet_3L(oxy, f0 = f_tanh, f1 = f_tanh)
            val (cols, rSq) = model_perc_2_0.forwardSelAll(0)
            println (s"rSq = $rSq")
            val k = cols.size
            val t = VectorD.range (1, k)
            new PlotM (t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"), "R^2 vs n for tanh tanh nn3l", lines = true)
            model_perc_2_0.train().eval()
            println(model_perc_2_0.report)

            val model_perc_2_1 = NeuralNet_3L(oxy, f0 = f_tanh, f1 = f_reLU)
            val (cols1, rSq1) = model_perc_2_1.forwardSelAll(0)
            println (s"rSq = $rSq1")
            val k1 = cols1.size
            val t1 = VectorD.range (1, k1)
            new PlotM (t1, rSq1.t, Array ("R^2", "R^2 bar", "R^2 cv"), "R^2 vs n for tanh relu nn3l", lines = true)
            model_perc_2_1.train().eval()
            println(model_perc_2_1.report)
        }
        //id activation function
        if(id_) {
            val model_perc_3_0 = NeuralNet_3L(oxy, f0 = f_id)
            model_perc_3_0.forwardSelAll(0)
//          no graph shown because of NaN values
            model_perc_3_0.train().eval()
            println(model_perc_3_0.report)
        }
    }

    if (NN_XL) {
        //sigmoid activation function in first layer, other activation functions in later layers
        if(sigmoid_) {
            val funcs1 = Array(f_sigmoid, f_reLU, f_sigmoid)
            val funcs2 = Array(f_sigmoid, f_tanh, f_reLU)
            val model_perc_0__0 = NeuralNet_XL(oxy, af = funcs1)
            val (cols, rSq) = model_perc_0__0.forwardSelAll(0)
            println (s"rSq = $rSq")
            val k = cols.size
            val t = VectorD.range (1, k)
            new PlotM (t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"), "R^2 vs n for sig relu sig nn4l", lines = true)
            model_perc_0__0.train().eval()
            println(model_perc_0__0.report)

            val model_perc_0__1 = NeuralNet_XL(oxy, af = funcs2)
            val (cols1, rSq1) = model_perc_0__1.forwardSelAll(0)
            println (s"rSq = $rSq1")
            val k1 = cols1.size
            val t1 = VectorD.range (1, k1)
            new PlotM (t1, rSq1.t, Array ("R^2", "R^2 bar", "R^2 cv"), "R^2 vs n for sig tanh relu nn4l", lines = true)
            model_perc_0__1.train().eval()
            println(model_perc_0__1.report)
        }
        //reLU activation function in first layer, other activation functions in later layers
        if(reLU_) {
            val funcs2 = Array(f_reLU, f_sigmoid, f_tanh)
            val model_perc_0__0 = NeuralNet_XL(oxy, af = funcs2)
            model_perc_0__0.forwardSelAll(0)
//          no graph shown because of NaN values
            model_perc_0__0.train().eval()
            println(model_perc_0__0.report)
        }
        //tanh activation function in first layer, other activation functions in later layers
        if(tanh_) {
            val funcs4 = Array(f_tanh, f_tanh, f_sigmoid)
            val funcs5 = Array(f_tanh, f_reLU, f_tanh)
            val model_perc_0__0 = NeuralNet_XL(oxy, af = funcs4)
            val (cols, rSq) = model_perc_0__0.forwardSelAll(0)
            println (s"rSq = $rSq")
            val k = cols.size
            val t = VectorD.range (1, k)
            new PlotM (t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"), "R^2 vs n for tanh tanh sig nn4l", lines = true)
            model_perc_0__0.train().eval()
            println(model_perc_0__0.report)

            val model_perc_0__1 = NeuralNet_XL(oxy, af = funcs5)
            val (cols1, rSq1) = model_perc_0__1.forwardSelAll(0)
            println (s"rSq = $rSq1")
            val k1 = cols1.size
            val t1 = VectorD.range (1, k1)
            new PlotM (t1, rSq1.t, Array ("R^2", "R^2 bar", "R^2 cv"), "R^2 vs n for tanh relu tanh nn4l", lines = true)
            model_perc_0__1.train().eval()
            println(model_perc_0__1.report)
        }
        //id activation function in first layer, other activation functions in later layers
        if(id_) {
            val funcs6 = Array(f_id, f_sigmoid, f_sigmoid)
            val model_perc_0__0 = NeuralNet_XL(oxy, af = funcs6)
            model_perc_0__0.forwardSelAll(0)
//          no graph shown because of NaN values
            model_perc_0__0.train().eval()
            println(model_perc_0__0.report)
        }
    }

    if (ELM) {
        //sigmoid activation function
        if(sigmoid_) {
            val model_elm_0 = ELM_3L1(oxy, f0 = f_sigmoid)
            val (cols, rSq) = model_elm_0.forwardSelAll(0, true)
            println (s"rSq = $rSq")
            val k = cols.size
            val t = VectorD.range (1, k)
            new PlotM (t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"), "R^2 vs n for sig elm", lines = true)
            model_elm_0.train().eval()
            println(model_elm_0.report)
        }
        //reLU activation function
        if(reLU_) {
            val model_elm_1 = ELM_3L1(oxy, f0 = f_reLU)
            model_elm_1.forwardSelAll(0,true)
//          no graph shown because of NaN values
            model_elm_1.train().eval()
            println(model_elm_1.report)
        }
        //tanh activation function
        if(tanh_) {
            val model_elm_2 = ELM_3L1(oxy, f0 = f_tanh)
            val (cols, rSq) = model_elm_2.forwardSelAll(0,true)
            println (s"rSq = $rSq")
            val k = cols.size
            val t = VectorD.range (1, k)
            new PlotM (t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"), "R^2 vs n for tanh elm", lines = true)
            model_elm_2.train().eval()
            println(model_elm_2.report)
        }
        //id activation function
        if(id_) {
            val model_elm_3 = ELM_3L1(oxy, f0 = f_id)
            model_elm_3.forwardSelAll(0,true)
//          no graph shown because of NaN values
            model_elm_3.train().eval()
            println(model_elm_3.report)
        }
    }
}