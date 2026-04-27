module GON_MulticastController #(
    parameter ID_SIZE = `XID_BITS
)(
    input clk,
    input rst,

    // config id
    input set_id,
    input [ID_SIZE - 1:0] id_in,
    output logic [ID_SIZE - 1:0] id,

    // tag
    input [ID_SIZE - 1:0] tag,

    input valid_in,
    output logic valid_out,
    input ready_in,
    output logic ready_out
);

    /*
                  ID              Ready (in)  Enable (out)  [Value] (out)
                  ⬇                 ⬇            ⬆             ⬆
          ---------------------------|-------------|-------------|-------
         |      -------              |             |             |       |
         |     |  reg  |             |             |             |       |
         |      -------   ___________|             |           ----      |
         |        ⬇      |   ____                 |    ----> / 0 1 \    |
         |      ------    -->|    |                |   |      -- ^ --    |
         |     |  ==  | ---->|    |----------------*---       ⬆   |     |
         |      ------    |  |AND |                           0    |     |
         |        ⬆    ---->|____|                                |     |
         |      <Tag>  |  |                                        |     |
         |             |  |                                        |     |
          -------------|-------------------------------------------------
                       ⬆ ⬇                                       ⬆
                       |  Ready (out)                          [Value] (in)
                       Enable (in)
    */

    // if set_id is true, then set id = imput id, else id = id
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            id <= 0;
        end else begin
            if (set_id) begin
                id <= id_in;
            end else begin
                id <= id;
            end
        end
    end

    // setting signal
    always_comb begin
        valid_out = valid_in && (tag == id);
        ready_out = ready_in && (tag == id);
    end


endmodule
