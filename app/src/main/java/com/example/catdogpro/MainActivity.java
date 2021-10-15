package com.example.catdogpro;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.widget.ImageViewCompat;

import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import com.example.catdogpro.ml.Model;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.schema.Tensor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;

public class MainActivity extends AppCompatActivity {

    private Button predict,upload;
    private ImageView img;
    private static final int PICK_IMAGE_REQUEST=1;
    private Uri mImageUri;
    private Bitmap bitmap;
    private TextView tv;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
       predict=findViewById(R.id.scan);
       upload=findViewById(R.id.upload);
       img=findViewById(R.id.imageView);
       tv=findViewById(R.id.result);
       upload.setOnClickListener(new View.OnClickListener() {
           @Override
           public void onClick(View view) {
               Intent intent = new Intent();
               intent.setType("image/*");// by this it shows only images to file chooser
               intent.setAction(Intent.ACTION_GET_CONTENT);
               startActivityForResult(intent,PICK_IMAGE_REQUEST);
           }
       });
       predict.setOnClickListener(new View.OnClickListener() {
           @Override
           public void onClick(View view) {
               bitmap=Bitmap.createScaledBitmap(bitmap,64,64,true);

               try {
                   Model model = Model.newInstance(getApplicationContext());

                   // Creates inputs for reference.
                   TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 64, 64, 3}, DataType.FLOAT32);

                   TensorImage tensorImage = new TensorImage(DataType.FLOAT32);
                   tensorImage.load(bitmap);
                   ByteBuffer byteBuffer = tensorImage.getBuffer();


                   inputFeature0.loadBuffer(byteBuffer);

                   // Runs model inference and gets result.
                   Model.Outputs outputs = model.process(inputFeature0);
                   TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

                   // Releases model resources if no longer used.
                   model.close();
                   if(outputFeature0.getFloatArray()[0]>0.5){
                      tv.setText("Dog");
                   }else{
                       tv.setText("Cat");
                   }
                  // tv.setText(String.valueOf(outputFeature0.getFloatArray()[0]));
               } catch (IOException e) {
                   // TODO Handle the exception
               }
           }
       });
    }
    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if(requestCode==PICK_IMAGE_REQUEST && resultCode==RESULT_OK && data!=null && data.getData()!=null){
            mImageUri=data.getData();
            //     mImageView.setImageURI(mImageUri);
            //
            try{
                bitmap= MediaStore.Images.Media.getBitmap(getContentResolver(),mImageUri);
                img.setImageBitmap(bitmap);
            }catch(Exception e){
                e.printStackTrace();
            }
            //
        }
    }
}